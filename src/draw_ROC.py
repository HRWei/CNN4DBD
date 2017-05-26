# -*- coding: utf-8 -*-

import math
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

SimSun = matplotlib.font_manager.FontProperties(fname='C:/Windows/Fonts/simsun.ttc')

org=open('./pixelate/config/05211000/org_ROC.txt','r')
conv=open('./pixelate/config/05211000/conv_ROC.txt','r')
conv2=open('./pixelate/config/05211000/conv2_ROC.txt','r')
_conv=open('./pixelate/config/05211000/_conv_ROC.txt','r')
chl=open('./pixelate/config/05211000/chl_ROC.txt','r')

orgl=org.readlines()
convl=conv.readlines()
conv2l=conv2.readlines()
_convl=_conv.readlines()
chll=chl.readlines()

org_FP=[]
org_TP=[]
for l in orgl:
    data=l.split()
    org_FP.append(float(data[0]))
    org_TP.append(float(data[1]))

conv_FP=[]
conv_TP=[]
for l in convl:
    data=l.split()
    conv_FP.append(float(data[0]))
    conv_TP.append(float(data[1]))

conv2_FP=[]
conv2_TP=[]
for l in conv2l:
    data=l.split()
    conv2_FP.append(float(data[0]))
    conv2_TP.append(float(data[1]))

_conv_FP=[]
_conv_TP=[]
for l in _convl:
    data=l.split()
    _conv_FP.append(float(data[0]))
    _conv_TP.append(float(data[1]))

chl_FP=[]
chl_TP=[]
for l in chll:
    data=l.split()
    chl_FP.append(float(data[0]))
    chl_TP.append(float(data[1]))

plt.rcParams['font.serif']=['Times New Roman']
plt.rcParams['font.family']='serif'
plt.rcParams['axes.unicode_minus']=False

plt.figure(figsize=(6,6))
plt.subplot(111)
plt.plot(chl_FP,chl_TP,color='k',label=u"策略一")
plt.plot(conv_FP,conv_TP,color='#00dd00',label=u"策略二")
plt.plot(conv2_FP,conv2_TP,color='b',label=u"策略三")
plt.plot(_conv_FP,_conv_TP,color='r',label=u"倍增策略二")
plt.plot(org_FP,org_TP,color='#ffaa00',label=u"类LeNet网络")
plt.plot([0.,1.],[0.,1.],'0.8',ls='--',lw=0.5)
plt.axis('square')
plt.xlim(0.,1.)
plt.ylim(0.,1.)
plt.xlabel('False Positive Rate',fontsize='13')
plt.ylabel('True Positive Rate',fontsize='13')
plt.legend(prop=SimSun)
plt.grid(alpha=0.4)
plt.savefig("./pre2/ROC.png")
plt.show()
