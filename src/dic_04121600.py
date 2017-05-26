import os

label_dir="./pixelate/config/04121600/"
if not os.path.isdir(label_dir):
    os.makedirs(label_dir)
xy_ftrain=open(label_dir+"train.txt",'w')
xy_ftest=open(label_dir+"test.txt",'w')

raw_label_dir="./pixelate/config/04050700/"
ftrain=open(raw_label_dir+"train.txt",'r')
ftest=open(raw_label_dir+"test.txt",'r')

ltrain = ftrain.readlines()
ltest = ftest.readlines()

strg="./pixelate/img/BambooKerasDataV2/yz/"
lstrg=len(strg)
for l in ltrain:
    xy_ftrain.write(l[:-2]+strg+l[lstrg:])

for l in ltest:
    xy_ftest.write(l[:-2]+strg+l[lstrg:])

ftrain.close()
ftest.close()

xy_ftrain.close()
xy_ftest.close()
