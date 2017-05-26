import os
import h5py
import string
import cv2

data_dir="./pixelate/img"
xz_dir=data_dir+"/BambooKerasDataV2/xz/"
yz_dir=data_dir+"/BambooKerasDataV2/yz/"
hdf5_dir=data_dir+"/BambooKerasDataV2/hdf5/"

xz_path=os.listdir(xz_dir)
yz_path=os.listdir(yz_dir)

if not os.path.isdir(hdf5_dir):
    os.makedirs(hdf5_dir)

flag=True
for x1,x2 in zip(xz_path,yz_path):
    if x1 != x2:
        flag=False

count = float(len(xz_path))
c = 0
print "%.1f%%" % (c * 100 / count)

if flag:
    for idx in xz_path:
        h5file=hdf5_dir+idx[:6]+".h5"
        h5f=h5py.File(h5file,'w')
        h5f['xz']=cv2.imread(xz_dir+idx,0)
        h5f['yz']=cv2.imread(yz_dir+idx,0)
        h5f.close()

        c += 1
        prc= c*100/count
        if int(prc*10)>int((c-1)*1000/count):
            print "%.1f%%" % prc
    print "Done!"
else:
    print "Data Error!"
