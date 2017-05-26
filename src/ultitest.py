import unittest
import caffe
import numpy as np
import cv2
from HistVec.Hist import Hist
from ImgFlattenTest.ImgFlatten import ImgFlatten
from AccTest.AccTest import Acc


class TestLayerWithParam(unittest.TestCase):
    def setUp(self):
        net_file = "./pixelate/config/05021330/train.prototxt"
        self.net = caffe.Net(net_file, caffe.TRAIN)

    def test_flatten(self):
        self.net.forward()

if __name__=='__main__':
    unittest.main()

