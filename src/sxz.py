import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import cv2

xzpath="./pixelate/img/BambooKerasDataV2/xy/000002.png"
yzpath="./pixelate/img/BambooKerasDataV2/yz/000002.png"

plt.rcParams['font.serif']=['Times New Roman']
plt.rcParams['font.family']='serif'
plt.rcParams['axes.unicode_minus']=False

xz=cv2.imread(xzpath,cv2.IMREAD_GRAYSCALE)
yz=cv2.imread(yzpath,cv2.IMREAD_GRAYSCALE)

# plt.figure(figsize=(8,3))
# plt.subplot(121)
# plt.axis('square')
# plt.xlim(0,31)
# plt.ylim(31,0)
# plt.imshow(xz,cmap=plt.cm.YlOrRd)
# plt.colorbar()

# plt.subplot(122)
# plt.axis('square')
# plt.xlim(0,31)
# plt.ylim(31,0)
# plt.imshow(yz,cmap=plt.cm.YlOrRd)
# plt.colorbar()
# plt.savefig("./pre2/traj.png")
# plt.show()

# path="./pre2/bkg.png"
# img=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
# plt.figure(figsize=(6,6))
# ax=plt.subplot(111)
# plt.axis('square')
# plt.xlim(0,167)
# plt.ylim(167,0)
# ax.set_xticks([])
# ax.set_yticks([])
# ax.spines['right'].set_color('none')
# ax.spines['top'].set_color('none')
# ax.spines['bottom'].set_color('none')
# ax.spines['left'].set_color('none')
# plt.imshow(img,cmap=plt.cm.gray)
# plt.savefig("./pre2/bkget.png")
# plt.show()

img3=np.zeros((32,32,3),dtype=xz.dtype)
r_xz=np.zeros((32,32),dtype=xz.dtype)
for i in range(32):
    for j in range(32):
        r_xz[i,j]=xz[i,31-j]
img3[:,:,0]=yz[:,:]
#img3[:,:,2]=xz[:,:]

plt.figure(figsize=(6,12))
ax=plt.subplot(111)
plt.axis('equal')
plt.xlim(0,31)
plt.ylim(63,0)
ax.set_xticks([])
ax.set_yticks([])
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.spines['bottom'].set_color('none')
ax.spines['left'].set_color('none')
plt.imshow(xz,cmap=plt.cm.gray)
plt.savefig("./pre2/xy.png")
plt.show()
