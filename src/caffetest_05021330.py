# -*- coding: utf-8 -*-

import os
import caffe
from caffe import layers as L, params as P, proto, to_proto
from ImgFlattenTest.ImgFlatten import ImgFlatten
from AccTest.AccTest import Acc

root = './pixelate'
root += '/config/05091200'
solver_proto=root+'/solver.prototxt'

caffe.set_device(0)
caffe.set_mode_gpu()
solver = caffe.SGDSolver(solver_proto)
solver.solve()
