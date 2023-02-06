import chainer.links as L
from scipy.signal import convolve2d
import chainer
import cupy as cp
import cv2
import numpy as np

HPF = np.load('SRM_Kernels.npy') #  [5,5,1,30]
H = HPF.transpose((3,2,0,1)) # [30,1,5,5]
srm_conv = L.Convolution2D.from_params(W = H, b = None, stride=1, pad =2).to_gpu()
# output size: [batchsize, 30,256,256]

kernel_reward = np.array([[0.0625,0.0625,0.0625],[0.0625,0.5,0.0625],[0.0625,0.0625,0.0625]],dtype=np.float32)

kernel_reward = np.expand_dims(kernel_reward, axis=0)

kernel_reward = np.repeat(kernel_reward,30,axis=0)

kernel_reward = np.expand_dims(kernel_reward, axis=0) #[1,30,5,5]

reward_conv = L.Convolution2D.from_params(W = kernel_reward, b = None, stride=1, pad=1).to_gpu()

#output_size: [batch_size, 1, 256, 256]

def set_wet_cost(rho_p1, rho_n1, img):
    wet_cost = 10 ** 10
    rho_p1[np.isnan(rho_p1)] = wet_cost
    rho_n1[np.isnan(rho_n1)] = wet_cost
    rho_p1[rho_p1 > wet_cost] = wet_cost
    rho_n1[rho_n1 > wet_cost] = wet_cost

    rho_p1[img == 255] = wet_cost
    rho_n1[img == 0] = wet_cost

    return  rho_p1, rho_n1
        
def get_srm_distance(batch_cover, batch_stego):
    batch_cover = chainer.cuda.to_gpu(batch_cover)
    batch_stego = chainer.cuda.to_gpu(batch_stego)
    c_res = srm_conv(batch_cover).data
    s_res = srm_conv(batch_stego).data
    distance = np.mean((s_res-c_res)**2)
    return cp.asnumpy(distance)
    
def srm_chainer_reward(cover, stego, pre_stego, confidence_reward = None):
    
    diff = np.array(stego!=pre_stego).astype(np.float32)
    #print(np.mean(diff))
    cover = chainer.cuda.to_gpu(cover)
    stego = chainer.cuda.to_gpu(stego)
    pre_stego = chainer.cuda.to_gpu(pre_stego)
    c_res = srm_conv(cover).data
    #c_res = reward_conv(c_res).data

    s_res = srm_conv(stego).data
    #s_res = reward_conv(s_res).data

    p_res = srm_conv(pre_stego).data
    #p_res = reward_conv(p_res).data
    
    reward = (((cp.asnumpy(c_res)-cp.asnumpy(p_res))**2)-((cp.asnumpy(c_res)-cp.asnumpy(s_res))**2))/30
    reward = chainer.cuda.to_gpu(reward)
    reward = reward_conv(reward).data
    reward = cp.asnumpy(reward)
    reward = np.squeeze(reward, 1)
    print("srm reward:" + str(np.mean(reward)))
    if confidence_reward is not None:
        
        for i in range(reward.shape[0]):
            con_reward = (diff[i,0,:,:] * confidence_reward[i]).copy()
            
            reward[i,:,:] = reward[i,:,:] +con_reward

    return reward
              

def get_mask(batch_cost, rate):
    batch_mask = np.zeros([batch_cost.shape[0], batch_cost.shape[2], batch_cost.shape[3]])
    tmp_cost = np.zeros(batch_cost.shape[2]*batch_cost.shape[3])
    for i in range(batch_cost.shape[0]):
        cost = batch_cost[i,0,:,:]
        mask = np.zeros(cost.shape)
        
        tmp_cost[:] = cost.reshape((-1))
        tmp_cost.sort()

        threshold = tmp_cost[int(256*256*rate)]
        mask[cost<threshold] = 1
        batch_mask[i,:,:] = mask
    return batch_mask 
    

def gen_diff(batch_cover, batch_stego):
    batch_diff = batch_cover.astype(np.float32) - batch_stego.astype(np.float32)
    batch_diff[batch_diff==0] = 128
    batch_diff[batch_diff==-1] = 0
    batch_diff[batch_diff==1] = 255

    for i in range(batch_diff.shape[0]):
        cv2.imwrite(str(i)+'.png',batch_diff[i,0,:,:])

def ternary_entropyf(pP1, pM1):
    p0 = 1-pP1-pM1
    P = np.hstack((p0.flatten(), pP1.flatten(), pM1.flatten()))
    H = -P*np.log2(P)
    eps = 2.2204e-16
    H[P<eps] = 0
    H[P>1-eps] = 0
    return np.sum(H)

def calc_lambda(rho_p1, rho_m1, message_length, n):
    l3 = 1e+3
    m3 = float(message_length+1)
    iterations = 0
    while m3 > message_length:
        l3 = l3 * 2
        pP1 = (np.exp(-l3 * rho_p1)) / (1 + np.exp(-l3 * rho_p1) + np.exp(-l3 * rho_m1))
        pM1 = (np.exp(-l3 * rho_m1)) / (1 + np.exp(-l3 * rho_p1) + np.exp(-l3 * rho_m1))
        m3 = ternary_entropyf(pP1, pM1)

        iterations += 1
        if iterations > 10:
            return l3
    l1 = 0
    m1 = float(n)
    lamb = 0
    iterations = 0
    alpha = float(message_length)/n
    # limit search to 30 iterations and require that relative payload embedded
    # is roughly within 1/1000 of the required relative payload
    while float(m1-m3)/n > alpha/1000.0 and iterations<300:
        lamb = l1+(l3-l1)/2
        pP1 = (np.exp(-lamb*rho_p1))/(1+np.exp(-lamb*rho_p1)+np.exp(-lamb*rho_m1))
        pM1 = (np.exp(-lamb*rho_m1))/(1+np.exp(-lamb*rho_p1)+np.exp(-lamb*rho_m1))
        m2 = ternary_entropyf(pP1, pM1)
        if m2 < message_length:
            l3 = lamb
            m3 = m2
        else:
            l1 = lamb
            m1 = m2
    iterations = iterations + 1;
    return lamb

def embedding_simulator(batch_x, batch_rho_p1, batch_rho_m1, m, seed):
    batch_y = np.zeros(batch_x.shape)
    for i in range(batch_x.shape[0]):
        x = batch_x[i,0,:,:]
        rho_p1 = batch_rho_p1[i,0,:,:]
        rho_m1 = batch_rho_m1[i, 0, :, :]

        n = x.shape[0]*x.shape[1]
        lamb = calc_lambda(rho_p1, rho_m1, m, n)
        pChangeP1 = (np.exp(-lamb * rho_p1)) / (1 + np.exp(-lamb * rho_p1) + np.exp(-lamb * rho_m1));
        pChangeM1 = (np.exp(-lamb * rho_m1)) / (1 + np.exp(-lamb * rho_p1) + np.exp(-lamb * rho_m1));
        y = x.copy()
        #print("seed: "+str(seed[i]))
        np.random.seed(seed[i])
        randChange = np.random.rand(y.shape[0], y.shape[1])
        y[randChange < pChangeP1] = y[randChange < pChangeP1] + 1;
        y[(randChange >= pChangeP1) & (randChange < pChangeP1+pChangeM1)] = y[(randChange >= pChangeP1) & (randChange < pChangeP1+pChangeM1)] - 1;
        batch_y[i,0,:,:] = y
    return batch_y
    





