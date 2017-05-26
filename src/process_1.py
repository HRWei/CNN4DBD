import os
import string
import numpy as np
import matplotlib.pyplot as plt
import cv2

def project(img,evecs,bins):
    hist=[0]*bins
    m,n=img.shape
    vc=[[m-1,n-1],[m-1,0],[0,n-1]]
    x=0;
    for v in vc:
        xt=np.abs(np.dot(evecs[:,0],v))
        if np.abs(xt)>x:
            x=xt
    x=x/bins+0.001

    for i in range(m):
        for j in range(n):
            idx=int(np.abs(np.dot(evecs[:,0],[i,j]))/x)
            hist[idx]+=img[i,j]

    return hist

def pca(data):
    mean_data=np.mean(data,axis=0)
    data-=mean_data
    C=np.cov(np.transpose(data))
    
    evals,evecs=np.linalg.eig(C)
    idx=evals.argsort()
    idx=idx[::-1]
    evecs=evecs[:,idx]
    evals=evals[idx]

    x=np.dot(np.transpose(evecs),np.transpose(data))
    return x,evecs


def thrData(img):
    mean_p=0
    count=0
    for row in img:
        for pix in row:
            if pix>0:
                mean_p+=pix
                count+=1
    mean_p=mean_p*1.0/count
    ret,img_thr = cv2.threshold(img,mean_p/2,255,cv2.THRESH_BINARY)

    data=[]
    m,n=img_thr.shape
    for i in range(m):
        for j in range(n):
            if img_thr[i,j]==255:
                data.append([i,j])
    
    return data


fname = "110ca813-ed3b-4411-a23a-0fb60d533881"
img=cv2.imread("./img/"+fname+"/0.xz"+".png",0)
bins=8
data=thrData(img)

x,evecs=pca(data)
hist=project(img,evecs,bins)
plt.figure(figsize=(8,4))
plt.subplot(121)
plt.plot(x[0,:],x[1,:],".")
plt.axis("square")

#_data=np.transpose(data)
plt.subplot(122)
plt.plot(range(bins),hist,"-")
plt.show()

#idx=0
#infile=open("./dat/"+fname+".txt",'w')
#rootdir="./img/"+fname+"/"
#for file in os.listdir(rootdir):
    #print os.path.join(rootdir,file)


#while(os.path.exists("./img/"+fname+"/"+str(idx)+".xz"+".png")):
    #img_xz=cv2.imread("./img/"+fname+"/"+str(idx)+".xz"+".png",0)
    #img_yz=cv2.imread("./img/"+fname+"/"+str(idx)+".yz"+".png",0)
    #data_xz=thrData(img_xz)
    #data_yz=thrData(img_yz)
    #x,evecs_x=pca(data_xz)
    #y,evecs_y=pca(data_yz)
    #hist_x=project(img_xz,evecs_x,bins)
    #hist_y=project(img_yz,evecs_y,bins)
    #hist=hist_x+hist_y
    #infile.write(" ".join(map(str,hist)))

    #idx+=1
    #print idx

#infile.close()