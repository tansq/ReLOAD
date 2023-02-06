import chainer
from chainer import Variable
import chainer.links as L
import chainer.functions as F
import numpy as np
import math
import cv2

import chainerrl
from chainerrl.agents import a3c

class DilatedConvBlock(chainer.Chain):

    def __init__(self, d_factor):
        super(DilatedConvBlock, self).__init__(
            diconv=L.DilatedConvolution2D(in_channels=64, out_channels=64, ksize=3, stride=1, pad=d_factor, dilate=d_factor, nobias=False),
            #bn=L.BatchNormalization(64)
        )

        self.train = True

    def __call__(self, x):
        h = F.relu(self.diconv(x))
        #h = F.relu(self.bn(self.diconv(x)))
        return h


class UNet(chainer.Chain, a3c.A3CModel):
    def __init__(self, n_actions):
        super(UNet, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(3, 16, 3, 2, 1)
            self.norm1 = L.BatchNormalization(16)
            self.conv2 = L.Convolution2D(16, 32, 3, 2, 1)
            self.norm2 = L.BatchNormalization(32)
            self.conv3 = L.Convolution2D(32, 64, 3, 2, 1)
            self.norm3 = L.BatchNormalization(64)
            self.conv4 = L.Convolution2D(64, 128, 3, 2, 1)
            self.norm4 = L.BatchNormalization(128)
            self.conv5 = L.Convolution2D(128, 128, 3, 2, 1)
            self.norm5 = L.BatchNormalization(128)
            self.conv6 = L.Convolution2D(128, 128, 3, 2, 1)
            self.norm6 = L.BatchNormalization(128)
            self.conv7 = L.Convolution2D(128, 128, 3, 2, 1)
            self.norm7 = L.BatchNormalization(128)
            self.conv8 = L.Convolution2D(128, 128, 3, 2, 1)
            self.norm8 = L.BatchNormalization(128)
            '''
            Value Branch
            '''
            self.v_deconv1 = L.Deconvolution2D(128, 128, 5, 2, 2,outsize = (2,2))
            self.v_denorm1 = L.BatchNormalization(128)
            self.v_deconv2 = L.Deconvolution2D(256, 128, 5, 2, 2,outsize = (4,4))
            self.v_denorm2 = L.BatchNormalization(128)
            self.v_deconv3 = L.Deconvolution2D(256, 128, 5, 2, 2,outsize = (8,8))
            self.v_denorm3 = L.BatchNormalization(128)
            self.v_deconv4 = L.Deconvolution2D(256, 128, 5, 2, 2,outsize = (16,16))
            self.v_denorm4 = L.BatchNormalization(128)
            self.v_deconv5 = L.Deconvolution2D(256, 64, 5, 2, 2,outsize = (32,32))
            self.v_denorm5 = L.BatchNormalization(64)
            self.v_deconv6 = L.Deconvolution2D(128, 32, 5, 2, 2,outsize = (64,64))
            self.v_denorm6 = L.BatchNormalization(32)
            self.v_deconv7 = L.Deconvolution2D(64, 16, 5, 2, 2,outsize = (128,128))
            self.v_denorm7 = L.BatchNormalization(16)
            self.v_deconv8 = L.Deconvolution2D(32, 1, 5, 2, 2,outsize = (256,256))

            '''
            Policy Branch
            '''
            self.p_deconv1 = L.Deconvolution2D(128, 128, 5, 2, 2,outsize = (2,2))
            self.p_denorm1 = L.BatchNormalization(128)
            self.p_deconv2 = L.Deconvolution2D(256, 128, 5, 2, 2,outsize = (4,4))
            self.p_denorm2 = L.BatchNormalization(128)
            self.p_deconv3 = L.Deconvolution2D(256, 128, 5, 2, 2,outsize = (8,8))
            self.p_denorm3 = L.BatchNormalization(128)
            self.p_deconv4 = L.Deconvolution2D(256, 128, 5, 2, 2,outsize = (16,16))
            self.p_denorm4 = L.BatchNormalization(128)
            self.p_deconv5 = L.Deconvolution2D(256, 64, 5, 2, 2,outsize = (32,32))
            self.p_denorm5 = L.BatchNormalization(64)
            self.p_deconv6 = L.Deconvolution2D(128, 32, 5, 2, 2,outsize = (64,64))
            self.p_denorm6 = L.BatchNormalization(32)
            self.p_deconv7 = L.Deconvolution2D(64, 16, 5, 2, 2,outsize = (128,128))
            self.p_denorm7 = L.BatchNormalization(16)
            self.p_deconv8 = chainerrl.policies.SoftmaxPolicy(L.Deconvolution2D(32, n_actions, 5, 2, 2,outsize = (256,256)))


    def pi_and_v(self, x):
        h1 = F.leaky_relu(self.norm1(self.conv1(x)))
        h2 = F.leaky_relu(self.norm2(self.conv2(h1)))
        h3 = F.leaky_relu(self.norm3(self.conv3(h2)))
        h4 = F.leaky_relu(self.norm4(self.conv4(h3)))
        h5 = F.leaky_relu(self.norm5(self.conv5(h4)))
        h6 = F.leaky_relu(self.norm6(self.conv6(h5)))
        h7 = F.leaky_relu(self.norm7(self.conv7(h6)))
        h8 = F.leaky_relu(self.norm8(self.conv8(h7)))
        ''' Output values'''

        v_dh = F.relu(F.dropout(self.v_denorm1(self.v_deconv1(h8))))
        v_dh = F.relu(F.dropout(self.v_denorm2(self.v_deconv2(F.concat((v_dh, h7))))))
        v_dh = F.relu(F.dropout(self.v_denorm3(self.v_deconv3(F.concat((v_dh, h6))))))
        v_dh = F.relu(self.v_denorm4(self.v_deconv4(F.concat((v_dh, h5)))))
        v_dh = F.relu(self.v_denorm5(self.v_deconv5(F.concat((v_dh, h4)))))
        v_dh = F.relu(self.v_denorm6(self.v_deconv6(F.concat((v_dh, h3)))))
        v_dh = F.relu(self.v_denorm7(self.v_deconv7(F.concat((v_dh, h2)))))
        v_out = F.sigmoid(self.v_deconv8(F.concat((v_dh, h1))))






        ''' Output policy'''
        p_dh = F.relu(F.dropout(self.p_denorm1(self.p_deconv1(h8))))
        p_dh = F.relu(F.dropout(self.p_denorm2(self.p_deconv2(F.concat((p_dh, h7))))))
        p_dh = F.relu(F.dropout(self.p_denorm3(self.p_deconv3(F.concat((p_dh, h6))))))
        p_dh = F.relu(self.p_denorm4(self.p_deconv4(F.concat((p_dh, h5)))))
        p_dh = F.relu(self.p_denorm5(self.p_deconv5(F.concat((p_dh, h4)))))
        p_dh = F.relu(self.p_denorm6(self.p_deconv6(F.concat((p_dh, h3)))))
        p_dh = F.relu(self.p_denorm7(self.p_deconv7(F.concat((p_dh, h2)))))
        p_out = self.p_deconv8(F.concat((p_dh, h1)))

        return p_out, v_out
