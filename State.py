import numpy as np
import sys
import cv2

class State():
    def __init__(self, size, move_range):
        self.state = np.zeros((size[0],3,size[2],size[3]), dtype=np.float32)
        self.move_range = move_range
        self.alpha = 1.25

    def reset(self, stego, action, diff):
        for i in range(stego.shape[0]):
            self.state[i, 0, :, :] = stego[i, 0, :, :]
            self.state[i, 1, :, :] = diff[i,0,:,:]
            self.state[i, 2, :, :] = action[i,:,:]

  
    def step(self, act,rho_p1,rho_n1):
        neutral = (self.move_range - 1)/2
        move = act.astype(np.float32)
        move = move - neutral
        
        coef_p1 = np.ones((self.state.shape[0], self.state.shape[2], self.state.shape[3]), np.float32)
        coef_n1 = np.ones(coef_p1.shape, np.float32)
        for i in range(int(neutral+1)):
            if i == 0:
                continue
            coef_p1[move==i] = self.alpha*i
            coef_n1[move==-i] = self.alpha*i
        adj_rho_p1 = rho_p1 / coef_p1.astype(np.float32)
        adj_rho_n1 = rho_n1 / coef_n1.astype(np.float32)
        
        
        return adj_rho_p1,adj_rho_n1




