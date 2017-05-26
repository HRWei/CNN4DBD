import os

label_dir="./pixelate/config/04102000/"
if not os.path.isdir(label_dir):
    os.makedirs(label_dir)
xz_ftrain=open(label_dir+"xz_train.txt",'w')
yz_ftrain=open(label_dir+"yz_train.txt",'w')
xz_ftest=open(label_dir+"xz_test.txt",'w')
yz_ftest=open(label_dir+"yz_test.txt",'w')

raw_label_dir="./pixelate/config/04050700/"
ftrain=open(raw_label_dir+"train.txt",'r')
ftest=open(raw_label_dir+"test.txt",'r')

ltrain = ftrain.readlines()
ltest = ftest.readlines()

strg="./pixelate/img/BambooKerasDataV2/yz/"
lstrg=len(strg)
for l in ltrain:
    xz_ftrain.write(l)
    yz_ftrain.write(strg+l[lstrg:])

for l in ltest:
    xz_ftest.write(l)
    yz_ftest.write(strg+l[lstrg:])

ftrain.close()
ftest.close()

xz_ftrain.close()
yz_ftrain.close()
xz_ftest.close()
yz_ftest.close()
