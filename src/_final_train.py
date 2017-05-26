
import os
import numpy as np
import caffe
from caffe import layers as L, params as P, proto, to_proto
from ImgFlattenTest.ImgFlatten import ImgFlatten

root = './pixelate'
root += '/config/04251400'
train_list='./pixelate/config/04191600/train.txt'
test_list='./pixelate/config/04191600/test.txt'
train_proto=root+'/train.prototxt'
test_proto=root+'/test.prototxt'
solver_proto=root+'/solver.prototxt'

loss_log=root+'/loss.txt'
acc_log=root+'/acc.txt'

max_iter = 500
test_interval = 10

caffe.set_device(0)
caffe.set_mode_gpu()
solver = caffe.SGDSolver(solver_proto)

train_loss = np.zeros(max_iter)
test_acc = np.zeros(int(np.ceil(max_iter / test_interval)))

# the main solver loop
for it in range(max_iter):
    print 'Iteration', it,'training...',
    solver.step(1)
    
    # store the train loss
    train_loss[it]=solver.net.blobs['SoftmaxWithLoss1'].data
    #solver.test_nets[0].forward(start='ImageData1')
    print 'loss',train_loss[it]

    if it % test_interval == 0:
        print 'Iteration', it, 'testing...',
        solver.test_nets[0].forward(start='ImageData1')
        acc=solver.test_nets[0].blobs['Accuracy1'].data
        print 'accuracy:',acc,'loss',train_loss[it]
        test_acc[it // test_interval] = acc


#floss=open(loss_log,'w')
#facc=open(acc_log,'w')

#floss.close()
#fass.close()
