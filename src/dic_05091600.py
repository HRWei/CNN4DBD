import os
import random

data_dir="./pixelate/img"
# sig_xz_dir=data_dir+"/Signal/xz/"
# bkg_xz_dir=data_dir+"/Background/xz/"
# sig_path=os.listdir(sig_xz_dir)
# bkg_path=os.listdir(bkg_xz_dir)
xz_dir=data_dir+"/BambooKerasDataV2/xy/"
path=os.listdir(xz_dir)


label_dir="./pixelate/config/05091600/"
if not os.path.isdir(label_dir):
    os.makedirs(label_dir)
ftrain=open(label_dir+"train.txt",'w')
ftest=open(label_dir+"test.txt",'w')

flabel=open("./pixelate/dat/BambooKerasDataV2.txt")
lines = flabel.readlines()
flabel.close()

tag_sig=[]
tag_bkg_1=[]
tag_bkg_2=[]
num_sig=0
num_bkg_1=0
num_bkg_2=0
for i in lines:
    split_tag=i.split()
    if split_tag[1]=="0":
        num_sig+=1
        tag_sig.append(int(split_tag[0]))
        path[int(split_tag[0])]+=" 0"
    elif split_tag[1]=="1":
        num_bkg_1+=1
        tag_bkg_1.append(int(split_tag[0]))
        path[int(split_tag[0])]+=" 1"
    else:
        num_bkg_2+=1
        tag_bkg_2.append(int(split_tag[0]))
        path[int(split_tag[0])]+=" 2"

print num_sig,num_bkg_1,num_bkg_2

count=15000
list_sig=random.sample(tag_sig,count/3)
list_bkg_1=random.sample(tag_bkg_1,count/3)
list_bkg_2=random.sample(tag_bkg_2,count/3)
train_tag=list_sig+list_bkg_1+list_bkg_2
random.shuffle(train_tag)
print len(train_tag)
for i in train_tag:
    ftrain.write(xz_dir+path[i]+"\n")

print "train txt completed"

_tag_sig=[]
_tag_bkg_1=[]
_tag_bkg_2=[]
count=1500
for i in tag_sig:
    if i not in list_sig:
        _tag_sig.append(i)
for i in tag_bkg_1:
    if i not in list_bkg_1:
        _tag_bkg_1.append(i)
for i in tag_bkg_2:
    if i not in list_bkg_2:
        _tag_bkg_2.append(i)
test_tag=random.sample(_tag_sig,count/3)+random.sample(_tag_bkg_1,count/3)+random.sample(_tag_bkg_2,count/3)
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
