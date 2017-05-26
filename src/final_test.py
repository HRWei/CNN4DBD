
import os
import numpy as np
import caffe
from caffe import layers as L, params as P, proto, to_proto
from ImgFlattenTest.ImgFlatten import ImgFlatten

root = './pixelate'
root += '/config/05211000'
test_list=root+'/test.txt'
fimg=open(test_list)
lines = fimg.readlines()
fimg.close()

def roc(nettype,blobnum,lines,root):

    deploy_proto=root+'/'+nettype+'_deploy.prototxt'
    modelFile=root+'/'+nettype+'_iter_5000.caffemodel'

    caffe.set_device(0)
    caffe.set_mode_gpu()
    net = caffe.Net(deploy_proto, modelFile, caffe.TEST)

    if blobnum==1:
        transformer = caffe.io.Transformer({'Data': net.blobs['Data'].data.shape})
        transformer.set_transpose('Data', (2,0,1))
    else:
        transformer = caffe.io.Transformer({'ImageData1': net.blobs['ImageData1'].data.shape,'ImageData3': net.blobs['ImageData3'].data.shape})
        transformer.set_transpose('ImageData1', (2,0,1))
        transformer.set_transpose('ImageData3', (2,0,1))


    size=100
    f=open(root+'/'+nettype+'_ROC.txt','w')
    for i in range(size):
        print '['+nettype+']',i*100/size,"%"
        true_sig=0
        false_sig=0
        for imgpath in lines:
            if blobnum==1:
                im=caffe.io.load_image(imgpath[:-2])
                net.blobs['Data'].data[0,0,...] = transformer.preprocess('Data',im)[0,...]
            else:
                xz_im=caffe.io.load_image(imgpath[:-16]+'xz'+imgpath[-14:-2])
                yz_im=caffe.io.load_image(imgpath[:-16]+'yz'+imgpath[-14:-2])
                net.blobs['ImageData1'].data[0,0,...] = transformer.preprocess('ImageData1',xz_im)[0,...]
                net.blobs['ImageData3'].data[0,0,...] = transformer.preprocess('ImageData3',yz_im)[0,...]

            out = net.forward()

            if net.blobs['prob'].data[0,0]>1.0*i/size:
                if int(imgpath[-2:])==0:
                    true_sig+=1
                else:
                    false_sig+=1
        
        f.write(str(false_sig/1000.0)+' '+str(true_sig/1000.0)+'\n')

    f.close()
    print '['+nettype+']',"Done!"

roc('org',1,lines,root)
roc('conv',1,lines,root)
roc('_conv',1,lines,root)
roc('conv2',2,lines,root)
roc('chl',1,lines,root)
