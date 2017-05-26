import os
from google.protobuf import text_format
import caffe.draw as draw
from caffe.proto import caffe_pb2

root = './pixelate'
root += '/config/05211000'
train_proto=root+'/conv.prototxt'

net = caffe_pb2.NetParameter()
text_format.Merge(open(train_proto).read(), net)
draw.draw_net_to_file(net,root+"/final_conv_train.png",'BT')
