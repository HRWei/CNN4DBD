import os
import random

data_dir="./pixelate/img"
# sig_xz_dir=data_dir+"/Signal/xz/"
# bkg_xz_dir=data_dir+"/Background/xz/"
# sig_path=os.listdir(sig_xz_dir)
# bkg_path=os.listdir(bkg_xz_dir)
xz_dir=data_dir+"/BambooKerasDataV2/xy/"
path=os.listdir(xz_dir)


label_dir="./pixelate/config/05211000/"
if not os.path.isdir(label_dir):
    os.makedirs(label_dir)
ftrain=open(label_dir+"train.txt",'w')
ftest=open(label_dir+"test.txt",'w')

flabel=open("./pixelate/dat/BambooKerasDataV2.txt")
lines = flabel.readlines()
flabel.close()

tag_sig=[]
tag_bkg=[]
num_sig=0
num_bkg=0
for i in lines:
    split_tag=i.split()
    if split_tag[1]=="0":
        num_sig+=1
        tag_sig.append(int(split_tag[0]))
        path[int(split_tag[0])]+=" 0"
    else:
        num_bkg+=1
        tag_bkg.append(int(split_tag[0]))
        path[int(split_tag[0])]+=" 1"

print num_sig,num_bkg

count=20000
list_sig=random.sample(tag_sig,count/2)
list_bkg=random.sample(tag_bkg,count/2)
train_tag=list_sig+list_bkg
random.shuffle(train_tag)
print len(train_tag)
for i in train_tag:
    ftrain.write(xz_dir+path[i]+"\n")

print "train txt completed"

_tag_sig=[]
_tag_bkg=[]
count=2000
for i in tag_sig:
    if i not in list_sig:
        _tag_sig.append(i)
for i in tag_bkg:
    if i not in list_bkg:
        _tag_bkg.append(i)
test_tag=random.sample(_tag_sig,count/2)+random.sample(_tag_bkg,count/2)
random.shuffle(test_tag)
for i in test_tag:
    ftest.write(xz_dir+path[i]+"\n")

ftrain.close()
ftest.close()

# for i in range(2000):
#     if random.randint(0,2):
#         ftrain.write(bkg_xz_dir+bkg_path[num_bkg]+" 0\n")
#         num_bkg+=1
#     else:
#         ftrain.write(sig_xz_dir+sig_path[num_sig]+" 1\n")
#         num_sig+=1

# for isig in sig_path[num_sig:]:
#     ftest.write(sig_xz_dir+sig_path[num_sig]+" 1\n")

# for ibkg in bkg_path[num_bkg:]:
#     ftest.write(bkg_xz_dir+bkg_path[num_bkg]+" 0\n")


# sig_yz_dir=data_dir+"/Signal/yz/"
# bkg_yz_dir=data_dir+"/Background/yz/"
# sig_path=os.listdir(sig_yz_dir)
# bkg_path=os.listdir(bkg_yz_dir)
# num_sig=0
# num_bkg=0
# for i in range(2000):
#     if random.randint(0,2):
#         ftrain.write(bkg_yz_dir+bkg_path[num_bkg]+" 0\n")
#         num_bkg+=1
#     else:
#         ftrain.write(sig_yz_dir+sig_path[num_sig]+" 1\n")
#         num_sig+=1

# for isig in sig_path[num_sig:]:
#     ftest.write(sig_yz_dir+sig_path[num_sig]+" 1\n")

# for ibkg in bkg_path[num_bkg:]:
#     ftest.write(bkg_yz_dir+bkg_path[num_bkg]+" 0\n")

print "Done!"
