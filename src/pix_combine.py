import os
import string
import numpy as np
import cv2

path="./pixelate/img/BambooKerasDataV2"
xlist=os.listdir(path+"/xz/")

count=len(xlist)
c=0.0
print "%.1f%%" % c
for name in xlist:

    xz=cv2.imread(path+"/xz/"+name)
    yz=cv2.imread(path+"/yz/"+name)
    cv2.imwrite(path+"/xy/"+name, np.vstack((xz,yz)))

    c += 1
    prc= c*100/count
    if int(prc*10)>int((c-1)*1000/count):
        print "%.1f%%" % prc

# xz=cv2.imread(path+"/xz/366423.png")
# yz=cv2.imread(path+"/yz/366423.png")
# cv2.imwrite(path+"/xy/366423.png", np.vstack((xz,yz)))
print "Done"
