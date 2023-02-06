import os

from mini_batch_loader import *
from chainer import serializers
from MyFCN import *
from chainer import cuda, optimizers, Variable
import sys
import math
import time
import chainerrl
import State
from scipy import io

from pixelwise_a3c import *
from utils import *
import argparse


#_/_/_/ paths _/_/_/ 
TRAINING_DATA_PATH          = "alaska1_train.txt" # alaska1_train.txt is composed of the path of images in the training dataset of ALASKA1
TESTING_DATA_PATH           = "samples.txt" # samples.txt is composed of the path of images in ./smaples
IMAGE_DIR_PATH              = "./samples/cover/" 

STEGO_PATH = ""
STEGO_DIR = ""
 
#_/_/_/ training parameters _/_/_/ 
LEARNING_RATE    = 0.001
TRAIN_BATCH_SIZE = 32
TEST_BATCH_SIZE  = 1 #must be 1
N_EPISODES           = 3000000
EPISODE_LEN = 6
GAMMA = 0.95 # discount factor

#noise setting
MEAN = 0
SIGMA = 15

N_ACTIONS = 3
MOVE_RANGE = 3
CROP_SIZE = 256

GPU_ID = 0

def a3c_test(loader, agent, payload, ear):

    test_data_size = MiniBatchLoader.count_paths(TESTING_DATA_PATH)
    current_state = State.State((TEST_BATCH_SIZE,1,CROP_SIZE,CROP_SIZE), MOVE_RANGE)


    for i in range(0, test_data_size, TEST_BATCH_SIZE):
        
        print(str(i)+" / 10000")
        raw_x, path, seed = loader.load_testing_data(np.array(range(i, i+TEST_BATCH_SIZE)))
        print(path) 
        stego_path = path[0].replace('cover/', STEGO_PATH)
        
        raw_x = raw_x.astype(np.float32)
        
        distortion = np.zeros(raw_x.shape).astype(np.float32)
        distortion_path = path[0].replace('cover','Rho_HILL'+str(payload))
        distortion_path = distortion_path.replace('pgm','mat')
        distortion[0,0,:,:] = io.loadmat(distortion_path)['rho']
        mask = get_mask(distortion,ear)

        m = payload * raw_x.shape[2] * raw_x.shape[3]
        action = np.ones((raw_x.shape[0],raw_x.shape[2],raw_x.shape[3]))
        
        init_cost_p1 = distortion.copy()
        init_cost_n1 = distortion.copy()
        init_cost_p1, init_cost_n1 = set_wet_cost(init_cost_p1, init_cost_n1, raw_x)

        init_stego = embedding_simulator(raw_x, init_cost_p1, init_cost_n1, m, seed).astype(np.float32)
        diff = raw_x - init_stego
        
        current_state.reset(init_stego/255.0,action,diff)
        srm_distance = get_srm_distance(raw_x, init_stego)
        lowest_distance = srm_distance
        print("distance between init stego and cover:" + str(np.mean(srm_distance)))

        current_stego = init_stego.copy()
        current_cost_p1 = distortion.copy()
        current_cost_n1 = distortion.copy()
        best_stego = current_stego.copy()
        
        for t in range(0, 99):
            print("iter:"+str(t))
            pre_stego = current_stego.copy().astype(np.float32)
            
            action = agent.act(current_state.state)
            action[mask==0] = 1
            
            current_cost_p1[:, 0, :, :], current_cost_n1[:, 0, :, :] = current_state.step(action, current_cost_p1[:,0,:,:], current_cost_n1[:,0,:,:])
            
            current_cost_p1, current_cost_n1 = set_wet_cost(current_cost_p1, current_cost_n1, raw_x)

            current_stego = embedding_simulator(raw_x, current_cost_p1, current_cost_n1, m, seed).astype(np.float32)
            
            current_diff = raw_x - current_stego
            
            current_state.state[:,0,:,:] = current_stego[:,0,:,:]/255.0
            current_state.state[:,1,:,:] = current_diff[:,0,:,:]
            current_state.state[:,2,:,:] = action[:,:,:]
                        
            srm_distance = np.mean(get_srm_distance(raw_x, current_stego))
            if srm_distance > lowest_distance:
                break
            lowest_distance = srm_distance
            best_stego = current_stego.copy()
            
            
            
        print("lowest distance: "+str(lowest_distance))
        
        print("Writing stego image: "+stego_path)
        cv2.imwrite(stego_path, best_stego[0,0,:,:])
        agent.stop_episode()



    
    
    
 
 
def main():
    #_/_/_/ load dataset _/_/_/ 
    parser = argparse.ArgumentParser()
    parser.add_argument('--payload', '-p', default=0.4, type=float, help='embedding payload')
    parser.add_argument('--ear', '-m', default=0.18, type=float, help='mask rate')
    parser.add_argument('--model_path', '-path', default="./model/ReLOAD_HILL_04.npz", type=str, help='path of trained ReLOAD model')
    args = parser.parse_args()
    payload = args.payload
    ear = args.ear
    model_path = args.model_path
    print(model_path) 
    global STEGO_DIR
    STEGO_DIR = "./samples/stego_"+"ReLOAD_HILL_m_"+str(ear)+"_p_"+str(payload)+"/"
    if not os.path.exists(STEGO_DIR):
        os.mkdir(STEGO_DIR)
    global STEGO_PATH 
    STEGO_PATH = "stego_"+"ReLOAD_HILL_m_"+str(ear)+"_p_"+str(payload)+"/"
    mini_batch_loader = MiniBatchLoader(
        TRAINING_DATA_PATH, 
        TESTING_DATA_PATH, 
        IMAGE_DIR_PATH, 
        CROP_SIZE)
 
    chainer.cuda.get_device_from_id(GPU_ID).use()
    

    # load myfcn model
    model = UNet(N_ACTIONS)
 
    #_/_/_/ setup _/_/_/
    optimizer = chainer.optimizers.Adam(alpha=LEARNING_RATE)
    optimizer.setup(model)

    agent = PixelWiseA3C(model, optimizer, EPISODE_LEN, GAMMA)

    chainer.serializers.load_npz(model_path, agent.model)
    agent.act_deterministically = True
    agent.model.to_gpu()
    
    a3c_test(mini_batch_loader, agent, payload, ear)
    
     
 
if __name__ == '__main__':
    try:

        start = time.time()
        main()
        end = time.time()
        print("{s}[s]".format(s=end - start))
        print("{s}[m]".format(s=(end - start)/60))
        print("{s}[h]".format(s=(end - start)/60/60))
        
    except Exception as error:
        print(error.message)
