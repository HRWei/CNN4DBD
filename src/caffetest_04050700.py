# -*- coding: utf-8 -*-

import os
import caffe
from caffe import layers as L, params as P, proto, to_proto

root = './pixelate'
root += '/config/04050700'
train_list=root+'/train.txt'
test_list=root+'/test.txt'
train_proto=root+'/train.prototxt'
test_proto=root+'/test.prototxt'
solver_proto=root+'/solver.prototxt'


def Lenet(img_list,batch_size,include_acc=False):

    data, label = L.ImageData(source=img_list, batch_size=batch_size, ntop=2, is_color=False,
        transform_param=dict(scale= 0.00390625))

    conv1=L.Convolution(data, kernel_size=5, stride=1,num_output=6, pad=0,weight_filler=dict(type='xavier'))
    pool1=L.Pooling(conv1, pool=P.Pooling.MAX, kernel_size=2, stride=2)
    conv2=L.Convolution(pool1, kernel_size=5, stride=1,num_output=16, pad=0,weight_filler=dict(type='xavier'))
    pool2=L.Pooling(conv2, pool=P.Pooling.MAX, kernel_size=2, stride=2)

    fc3=L.InnerProduct(pool2, num_output=100,weight_filler=dict(type='xavier'))
    relu3=L.ReLU(fc3, in_place=True)
    fc4 = L.InnerProduct(relu3, num_output=2,weight_filler=dict(type='xavier'))
    loss = L.SoftmaxWithLoss(fc4, label)
    
    if include_acc:
        acc = L.Accuracy(fc4, label)
        return to_proto(loss, acc)
    else:
        return to_proto(loss)

def newmethod521():
    return ImageData
    
def write_net():
    with open(train_proto, 'w') as f:
        f.write(str(Lenet(train_list,batch_size=16)))
    with open(test_proto, 'w') as f:
        f.write(str(Lenet(test_list,batch_size=2000, include_acc=True)))


def gen_solver(solver_file,train_net,test_net):
    s=proto.caffe_pb2.SolverParameter()
    s.train_net =train_net
    s.test_net.append(test_net)
    
    s.test_interval = 625    #测试间隔参数：训练完一次所有的图片，进行一次测试  
    s.test_iter.append(1)   #测试迭代次数，需要迭代8次，才完成一次所有数据的测试
    s.max_iter = 1250       #2 epochs，最大训练次数
    s.base_lr = 0.01    #基础学习率
    s.momentum = 0.9    #动量
    s.weight_decay = 5e-4  #权值衰减项
    s.lr_policy = 'step'   #学习率变化规则
    s.stepsize=3000         #学习率变化频率
    s.gamma = 0.1          #学习率变化指数
    s.display = 50         #屏幕显示间隔
    s.snapshot = 625       #保存caffemodel的间隔
    s.snapshot_prefix = root+'/lenet'   #caffemodel前缀
    s.type = 'SGD'         # 优化算法：随机梯度下降
    s.solver_mode = proto.caffe_pb2.SolverParameter.GPU    # 加速
    # 写入solver.prototxt
    with open(solver_file, 'w') as f:
        f.write(str(s))


def training(solver_proto):
    caffe.set_device(0)
    caffe.set_mode_gpu()
    solver = caffe.SGDSolver(solver_proto)
    solver.solve()

if __name__ == '__main__':
    if not os.path.isdir(root):
        os.makedirs(root)
    write_net()
    gen_solver(solver_proto, train_proto, test_proto)
    training(solver_proto)
