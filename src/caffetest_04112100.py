# -*- coding: utf-8 -*-

import os
import caffe
from caffe import layers as L, params as P, proto, to_proto

root = './pixelate'
root += '/config/04112100'
xz_train_list=root+'/xz_train.txt'
yz_train_list=root+'/yz_train.txt'
xz_test_list=root+'/xz_test.txt'
yz_test_list=root+'/yz_test.txt'
train_proto=root+'/train.prototxt'
test_proto=root+'/test.prototxt'
solver_proto=root+'/solver.prototxt'


def Lenet(xz_img_list,yz_img_list,batch_size,include_acc=False):

    data1, label1 = L.ImageData(source=xz_img_list, batch_size=batch_size, ntop=2, is_color=False,
        transform_param=dict(scale= 0.00390625))
    data2, label2 = L.ImageData(source=yz_img_list, batch_size=batch_size, ntop=2, is_color=False,
        transform_param=dict(scale= 0.00390625))

    conv1=L.Convolution(data1, kernel_size=5, stride=1,num_output=6, pad=0,weight_filler=dict(type='xavier'))
    pool1=L.Pooling(conv1, pool=P.Pooling.MAX, kernel_size=2, stride=2)
    conv2=L.Convolution(pool1, kernel_size=5, stride=1,num_output=16, pad=0,weight_filler=dict(type='xavier'))
    pool2=L.Pooling(conv2, pool=P.Pooling.MAX, kernel_size=2, stride=2)

    conv3=L.Convolution(data2, kernel_size=5, stride=1,num_output=6, pad=0,weight_filler=dict(type='xavier'))
    pool3=L.Pooling(conv3, pool=P.Pooling.MAX, kernel_size=2, stride=2)
    conv4=L.Convolution(pool3, kernel_size=5, stride=1,num_output=16, pad=0,weight_filler=dict(type='xavier'))
    pool4=L.Pooling(conv4, pool=P.Pooling.MAX, kernel_size=2, stride=2)

    fc1=L.InnerProduct(pool2, num_output=100,weight_filler=dict(type='xavier'))
    fc2=L.InnerProduct(pool4, num_output=100,weight_filler=dict(type='xavier'))
    concat=L.Concat(fc1,fc2)
    relu3=L.ReLU(concat, in_place=True)
    fc4 = L.InnerProduct(relu3, num_output=2,weight_filler=dict(type='xavier'))
    loss = L.SoftmaxWithLoss(fc4, label1)
    
    if include_acc:
        acc = L.Accuracy(fc4, label1)
        return to_proto(loss, acc)
    else:
        return to_proto(loss)

def newmethod521():
    return ImageData
    
def write_net():
    with open(train_proto, 'w') as f:
        f.write(str(Lenet(xz_train_list,yz_train_list,batch_size=16)))
    with open(test_proto, 'w') as f:
        f.write(str(Lenet(xz_test_list,yz_test_list,batch_size=2000, include_acc=True)))


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
