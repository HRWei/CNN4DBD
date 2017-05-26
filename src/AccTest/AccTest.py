import unittest
import caffe
import numpy as np

class Acc(caffe.Layer):

    def setup(self, bottom, top):
        assert len(bottom) == 2,            'requires two layer.bottom'
        assert len(top) == 3,               'requires two layer.top'

        # store input as class variables
        top[0].reshape(1)
        top[1].reshape(1)
        top[2].reshape(3)


    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        actual_sig=0.0
        true_sig=0.0
        for h in range(bottom[0].data.shape[0]):
            if bottom[0].data[h,0]>bottom[0].data[h,1]:
                actual_sig+=1
                if bottom[1].data[h]==0:
                    true_sig+=1
        top[0].data[0]=true_sig/actual_sig

        pred_sig=0.0
        true_sig=0.0
        for h in range(bottom[0].data.shape[0]):
            if bottom[1].data[h]==0:
                pred_sig+=1
                if bottom[0].data[h,0]>bottom[0].data[h,1]:
                    true_sig+=1
        top[1].data[0]=true_sig/pred_sig
        top[2].data[...]=[true_sig,actual_sig,pred_sig]

    def backward(self, top, propagate_down, bottom):
        pass
