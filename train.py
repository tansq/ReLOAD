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

from pixelwise_a3c import *
from scipy import io
from utils import *
import argparse


#_/_/_/ paths _/_/_/
TRAINING_DATA_PATH          = "./alaska1_train.txt" # alaska1_train.txt is composed of the path of images in the training dataset of ALASKA1
TESTING_DATA_PATH           = "./alaska1_val.txt" # alaska1_val.txt is composed of the path of images in the validating dataset of ALASKA1
IMAGE_DIR_PATH              = "" #your path of ALASKA1 dataset
SAVE_PATH = ""
STEGO_PATH = ""
STEGO_DIR = ""

#_/_/_/ training parameters _/_/_/
LEARNING_RATE    = 0.001
TRAIN_BATCH_SIZE = 32
TEST_BATCH_SIZE  = 1 #must be 1
N_EPISODES           = 10000
EPISODE_LEN = 6
SNAPSHOT_EPISODES  = 100
TEST_EPISODES = 2000
GAMMA = 0.95 # discount factor

#noise setting
MEAN = 0
SIGMA = 15

N_ACTIONS = 3 
MOVE_RANGE = 3
CROP_SIZE = 256

GPU_ID = 0

BEST_MODEL_EPOCH = 0

def a3c_test(loader, agent, ear, payload):

    test_data_size = MiniBatchLoader.count_paths(TESTING_DATA_PATH)
    current_state = State.State((TEST_BATCH_SIZE,1,CROP_SIZE,CROP_SIZE), MOVE_RANGE)
    sum_distance = 0
    for i in range(0, test_data_size, TEST_BATCH_SIZE):
        lowest_distance = 9999999
        print(str(i)+" / 2000")
        raw_x, path, seed = loader.load_testing_data(np.array(range(i, i+TEST_BATCH_SIZE)))
        
        raw_x = raw_x.astype(np.float32)
        distortion_path = path[0].replace('cover', 'MipodRhoP'+str(payload))
        distortion_path = distortion_path.replace('pgm','mat')
        distortion = np.zeros(raw_x.shape).astype(np.float32)
        distortion[0,0,:,:] = io.loadmat(distortion_path)['rho']
        mask = get_mask(distortion, ear) #mask:0 for ineffective agent, 1 for effective agent
        m = payload * raw_x.shape[2] * raw_x.shape[3]
        action = np.ones((raw_x.shape[0],raw_x.shape[2],raw_x.shape[3])) #action: 0 for embedding polarity of -1, 2 for embedding polarity of 1
        init_cost_p1 = distortion.copy()
        init_cost_n1 = distortion.copy()
        init_cost_p1, init_cost_n1 = set_wet_cost(init_cost_p1, init_cost_n1, raw_x)

        init_stego = embedding_simulator(raw_x, init_cost_p1, init_cost_n1, m, seed).astype(np.float32)
        diff = raw_x - init_stego
        # Initialize parameters for initial state
        current_state.reset(init_stego/255.0,action,diff)
        srm_distance = get_srm_distance(raw_x, init_stego)
        print("distance between init stego and cover:" + str(np.mean(srm_distance)))

        best_stego = init_stego.copy()

        current_stego = init_stego.copy()
        current_cost_p1 = distortion.copy()
        current_cost_n1 = distortion.copy()

        for t in range(0, 99):
            print("iter:"+str(t))
            pre_stego = current_stego.copy().astype(np.float32)

            action = agent.act(current_state.state)
            action[mask == 0] = 1
            current_cost_p1[:, 0, :, :], current_cost_n1[:, 0, :, :] = current_state.step(action, current_cost_p1[:,0,:,:], current_cost_n1[:,0,:,:])
            
            current_cost_p1, current_cost_n1 = set_wet_cost(current_cost_p1, current_cost_n1, raw_x)

            current_stego = embedding_simulator(raw_x, current_cost_p1, current_cost_n1, m, seed).astype(np.float32)
            current_diff = raw_x - current_stego
            #update state parameters
            current_state.state[:,0,:,:] = current_stego[:,0,:,:]/255.0
            current_state.state[:,1,:,:] = current_diff[:,0,:,:]
            current_state.state[:,2,:,:] = action[:,:,:]
            
            srm_distance = np.mean(get_srm_distance(raw_x, current_stego))
            if srm_distance > lowest_distance:
                break
            lowest_distance = srm_distance
            best_stego = current_stego.copy()
            best_diff = current_diff.copy()
        
        sum_distance+=lowest_distance
        print("testing final distance: "+ str(np.mean(lowest_distance)))
        print("+1: "+ str(np.mean(best_diff==1)) + " -1: "+str(np.mean(best_diff==-1)))

        agent.stop_episode()
    return sum_distance/ test_data_size





def main():
    #_/_/_/ load dataset _/_/_/
    TEST_CURRENT_DISTANCE = 99999999
    parser = argparse.ArgumentParser()
    parser.add_argument('--payload', '-p', default=0.4, type=float, help='embedding payload')
    parser.add_argument('--ear', '-m', default=0.18, type=float, help='mask rate')
    args = parser.parse_args()
    payload = args.payload
    ear = args.ear
    SAVE_PATH = "./model/ReLOAD_MiPOD_m_"+str(ear)+"_p_"+str(payload)+'/'
    mini_batch_loader = MiniBatchLoader(
        TRAINING_DATA_PATH,
        TESTING_DATA_PATH,
        IMAGE_DIR_PATH,
        CROP_SIZE)



    chainer.cuda.get_device_from_id(GPU_ID).use()

    current_state = State.State((TRAIN_BATCH_SIZE,1,CROP_SIZE,CROP_SIZE), MOVE_RANGE)

    # load myfcn model
    model = UNet(N_ACTIONS)

    #_/_/_/ setup _/_/_/

    optimizer = chainer.optimizers.Adam(alpha=LEARNING_RATE)
    optimizer.setup(model)
    
    agent = PixelWiseA3C(model, optimizer, EPISODE_LEN, GAMMA)
    agent.model.to_gpu()
    
    
    #_/_/_/ training _/_/_/

    train_data_size = MiniBatchLoader.count_paths(TRAINING_DATA_PATH)
    indices = np.random.permutation(train_data_size)
    i = 0
    for episode in range(1, N_EPISODES+1):
        # display current episode
        print("episode %d" % episode)

        # load images
        r = indices[i:i+TRAIN_BATCH_SIZE]
        raw_x, path, seed = mini_batch_loader.load_training_data(r)
        raw_x = raw_x.astype(np.float32)

        distortion = raw_x.copy()
        for batch_idx in range(distortion.shape[0]):
            rho_path = path[batch_idx].replace('cover','MipodRhoP'+str(payload))
            rho_path = rho_path.replace('pgm','mat')            
            distortion[batch_idx,0,:,:] = io.loadmat(rho_path)['rho']
        mask = get_mask(distortion, ear)
        m = payload * 256 * 256
        action = np.ones((raw_x.shape[0],raw_x.shape[2],raw_x.shape[3]))
        
        
        init_cost_p1 = distortion.copy()
        init_cost_n1 = distortion.copy()
        init_cost_p1, init_cost_n1 = set_wet_cost(init_cost_p1, init_cost_n1, raw_x)

        init_stego = embedding_simulator(raw_x, init_cost_p1, init_cost_n1, m, seed).astype(np.float32)
        diff = raw_x - init_stego
        # Initialize parameters for initial state
        current_state.reset(init_stego/255.0,action,diff)
        srm_distance = get_srm_distance(raw_x, init_stego)
        print("distance between init stego and cover:" + str(np.mean(srm_distance)))

        reward = np.zeros(action.shape, np.float32)
        sum_reward = 0
        current_stego = init_stego.copy().astype(np.float32)
        current_cost_p1 = distortion.copy()
        current_cost_n1 = distortion.copy()


        for t in range(0, EPISODE_LEN):
            print("Action: "+str(t))
            pre_stego = current_stego.copy()
            action = agent.act_and_train(current_state.state, reward)

            action[mask == 0] = 1

            current_cost_p1[:, 0, :, :], current_cost_n1[:, 0, :, :] = current_state.step(action, current_cost_p1[:,0,:,:], current_cost_n1[:,0,:,:])
            
            current_cost_p1, current_cost_n1 = set_wet_cost(current_cost_p1, current_cost_n1, raw_x)

            current_stego = embedding_simulator(raw_x, current_cost_p1, current_cost_n1, m, seed).astype(np.float32)

            current_diff = raw_x - current_stego
            #update state parameters
            current_state.state[:,0,:,:] = current_stego[:,0,:,:]/255.0
            current_state.state[:,1,:,:] = current_diff[:,0,:,:]
            current_state.state[:,2,:,:] = action[:,:,:]

            reward = srm_chainer_reward(raw_x, current_stego, pre_stego)
            
            sum_reward += np.mean(reward) * np.power(GAMMA, t)
        srm_distance = get_srm_distance(raw_x, current_stego)
        print("distance between current stego and cover:" + str(np.mean(srm_distance)))
        print("sum_reward:"+str(sum_reward))
        with open('sum_reward_ReLOAD_MiPOD_m_'+str(ear)+'_p_'+str(payload)+'.txt','a+') as f:
            f.write(str(sum_reward)+' '+str(srm_distance)+'\n')
        agent.stop_episode_and_train(current_state.state, reward, True)

        if episode % TEST_EPISODES == 0:
            #_/_/_/ testing _/_/_/
            current_distance = a3c_test(mini_batch_loader, agent, ear, payload)
            if current_distance<TEST_CURRENT_DISTANCE:
                TEST_CURRENT_DISTANCE = current_distance
                BEST_MODEL_EPOCH = episode

        if episode % SNAPSHOT_EPISODES == 0:
            agent.save(SAVE_PATH+str(episode))

        if i+TRAIN_BATCH_SIZE >= train_data_size:
            i = 0
            indices = np.random.permutation(train_data_size)
        else:
            i += TRAIN_BATCH_SIZE

        if i+2*TRAIN_BATCH_SIZE >= train_data_size:
            i = train_data_size - TRAIN_BATCH_SIZE

        optimizer.alpha = LEARNING_RATE*((1-episode/N_EPISODES)**0.9)
    print("Best model: " + str(BEST_MODEL_EPOCH))


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
