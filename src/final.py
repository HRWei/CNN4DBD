# -*- coding: utf-8 -*-

import os
import caffe
from caffe import layers as L, params as P, proto, to_proto
from ImgFlattenTest.ImgFlatten import ImgFlatten
from AccTest.AccTest3 import Acc

root = './pixelate'
root += '/config/05211000'
solver_proto=root+'/solver.prototxt'

caffe.set_device(0)
caffe.set_mode_gpu()
solver = caffe.SGDSolver(solver_proto)
solver.solve()

test_list=root+'/test.txt'
deploy_proto=root+'/conv_deploy.prototxt'
modelFile=root+'/conv_'+'_iter_5000.caffemodel'


net = caffe.Net(deploy_proto, modelFile, caffe.TEST)

fimg=open(test_list)
lines = fimg.readlines()
fimg.close()

transformer = caffe.io.Transformer({'Data': net.blobs['Data'].data.shape})
transformer.set_transpose('Data', (2,0,1))
# transformer = caffe.io.Transformer({'ImageData1': net.blobs['ImageData1'].data.shape,'ImageData3': net.blobs['ImageData3'].data.shape})
# transformer.set_transpose('ImageData1', (2,0,1))
# transformer.set_transpose('ImageData3', (2,0,1))

true_sig=0
actual_sig=0
pred_sig=0
count=len(lines)
c = 0
print "%d%%" % (c * 100 / count)

for imgpath in lines:
    # xz_im=caffe.io.load_image(imgpath[:-16]+'xz'+imgpath[-14:-2])
    # yz_im=caffe.io.load_image(imgpath[:-16]+'yz'+imgpath[-14:-2])
    # net.blobs['ImageData1'].data[0,0,...] = transformer.preprocess('ImageData1',xz_im)[0,...]
    # net.blobs['ImageData3'].data[0,0,...] = transformer.preprocess('ImageData3',yz_im)[0,...]
    im=caffe.io.load_image(imgpath[:-2])
    net.blobs['Data'].data[0,0,...] = transformer.preprocess('Data',im)[0,...]

    out = net.forward()

    if net.blobs['prob'].data[0,0]>0.5:
        actual_sig+=1
        if int(imgpath[-2:])==0:
            true_sig+=1
    if int(imgpath[-2:])==0:
        pred_sig+=1

    c += 1
    prc= c*100/count
    if prc>(c-1)*100/count:
        print "%d%%" % prc

print true_sig,actual_sig,pred_sig

