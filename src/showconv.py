import numpy as np
import matplotlib.pyplot as plt
import os
import caffe
import sys
import cv2

deployPrototxt =  "./pixelate/config/05211000/conv2_deploy.prototxt"
modelFile = "./pixelate/config/05211000/conv2_iter_5000.caffemodel"


def initilize():
    print 'initilize ... '
    caffe.set_mode_gpu()
    caffe.set_device(0)
    net = caffe.Net(deployPrototxt, modelFile, caffe.TEST)
    return net


def vis_square(data, padsize=1, padval=0 ):
    data -= data.min()
    data /= data.max()
    
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, int(n ** 2 - data.shape[0])), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))
    
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

    image=[]
    for row in data:
        image.append(np.array(row).flatten())

    plt.savefig("./pixelate/config/05211000/conv2(3).png")
    plt.imshow(image,cmap='gray')


net = initilize()
filters = net.params['Convolution3'][0].data
#with open('FirstLayerFilter.pickle','wb') as f:
#pickle.dump(filters,f)
vis_square(filters.transpose((0, 2, 3, 1)))
plt.show()
    # feat = net.blobs['conv1'].data[0, :36]
    # with open('FirstLayerOutput.pickle','wb') as f:
    #    pickle.dump(feat,f)
    # vis_square(feat,padval=1)
    # pool = net.blobs['pool1'].data[0,:36]
    # with open('pool1.pickle','wb') as f:
    #    pickle.dump(pool,f)
    # vis_square(pool,padval=1)
