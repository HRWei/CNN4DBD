import unittest
import caffe
import numpy as np
import cv2

class ImgFlatten(caffe.Layer):

    def setup(self, bottom, top):
        assert len(bottom) == 1,            'requires a single layer.bottom'
        assert len(top) == 1,               'requires a single layer.top'
        
        params = eval(self.param_str)

        # store input as class variables
        self.batch_size = bottom[0].data.shape[0]
        self.num_output = bottom[0].data.shape[1]

        # Create a batch loader to load the images.
        self.batch_loader = ImgLoader(params,bottom[0].data.shape)
        top[0].reshape(self.batch_size,self.num_output,self.batch_loader.height,self.batch_loader.width)
        print "FlattenImageLayerInfo", top[0].shape

    def reshape(self, bottom, top):      
        pass

    def forward(self, bottom, top):
        for h in range(self.batch_size):
            for d in range(self.num_output):
                top[0].data[h,d,...]=self.batch_loader.load_image(bottom[0].data[h,d,...])

    def backward(self, top, propagate_down, bottom):
        pass


class ImgLoader(object):

    def __init__(self, params,shape):
        self.input_shape=shape
        self.kernel_size = params['kernel_size']
        self.stride = params['stride']
        self.row_cnt=int((self.input_shape[-1] - self.kernel_size)/self.stride+1)
        self.width=self.row_cnt*self.kernel_size
        self.height=self.width*2

    def load_image(self,img):
        flat_img=np.zeros((self.height,self.width))
        for row in range(self.row_cnt):
            for col in range(self.row_cnt):
                flat_img[
                    row*self.kernel_size:(row+1)*self.kernel_size,
                    col*self.kernel_size:(col+1)*self.kernel_size
                    ]=img[
                        row*self.stride:row*self.stride+self.kernel_size,
                        col*self.stride:col*self.stride+self.kernel_size
                        ]
        for row in range(self.row_cnt,2*self.row_cnt):
            for col in range(self.row_cnt):
                flat_img[
                    row*self.kernel_size:(row+1)*self.kernel_size,
                    col*self.kernel_size:(col+1)*self.kernel_size
                    ]=img[
                        (row-self.row_cnt)*self.stride+self.input_shape[-1]:(row-self.row_cnt)*self.stride+self.input_shape[-1]+self.kernel_size,
                        col*self.stride:col*self.stride+self.kernel_size
                        ]

        return flat_img

class TestLayerWithParam(unittest.TestCase):
    def setUp(self):
        net_file = "./pixelate/config/04182100/train.prototxt"
        self.net = caffe.Net(net_file, caffe.TRAIN)

    def test_flatten(self):
        self.net.forward()

if __name__=='__main__':
    unittest.main()
