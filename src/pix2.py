import os
import string
import numpy as np
from xml.etree.cElementTree import *
import cv2


def pixelate(xzx, xzz, xzc, name, path="./img/"):
    min_xzx = min(xzx)
    max_xzx = max(xzx)
    min_xzz = min(xzz)
    max_xzz = max(xzz)

    _bin = 2
    min_xzx_std = int(min_xzx)
    max_xzx_std = int(max_xzx) + _bin
    num_cell_x = (max_xzx_std - min_xzx_std) / _bin

    min_xzz_std = int(min_xzz)
    max_xzz_std = int(max_xzz) + _bin
    num_cell_z = (max_xzz_std - min_xzz_std) / _bin

    num_cell = max([num_cell_x, num_cell_z])+2
    sum_xz = np.zeros((num_cell, num_cell))
    bias_x = (num_cell - num_cell_x) / 2
    bias_z = (num_cell - num_cell_z) / 2
    for x, z, c in zip(xzx, xzz, xzc):
        sum_xz[int((x - min_xzx_std) / _bin) + bias_x, int((z - min_xzz_std) /
                                                           _bin) + bias_z] += c

    max_xz = sum_xz.max()
    if max_xz == 0:
        return
    for i in range(num_cell):
        for j in range(num_cell):
            sum_xz[i, j] = sum_xz[i, j] * 255.0 / max_xz

    if not os.path.exists(path):
        os.makedirs(path)
    cv2.imwrite(path + name, cv2.resize(sum_xz, (32, 32)))


def dataSplit(data_s):
    xzx = []
    xzz = []
    xzc = []
    data_l = data_s.split()
    s = len(data_l)
    if s % 3 != 0 or s < 3:
        print "Error!"
        return [0], [0], [0]
    for i in range(len(data_l) / 3):
        xzx.append(string.atof(data_l[i * 3]))
        xzz.append(string.atof(data_l[i * 3 + 1]))
        xzc.append(string.atof(data_l[i * 3 + 2]))
    return xzx, xzz, xzc


print os.getcwd()
# fname= "14885421-4eba-4e66-bd4e-bb416df19b42"  # Background
# fn ="Background"
# fname = "110ca813-ed3b-4411-a23a-0fb60d533881"  # Signal
# fn ="Signal"
fname = "BambooKerasDataV2"
fn = "BambooKerasDataV2"
tree = parse("./pixelate/dat/" + fname + ".xml")
root = tree.getroot()
count = string.atof(root.get("COUNT"))
c = 0
print "%.1f%%" % (c * 100 / count)

for segnode in root.findall("segment"):
    xnode = segnode.find("xzc")
    xzx, xzz, xzc = dataSplit(xnode.text)
    pixelate(
        xzx,
        xzz,
        xzc,
        name=segnode.get("ID").zfill(6) + ".png",
        path="./pixelate/img/" + fn + "/xz/")
    ynode = segnode.find("yzc")
    yzy, yzz, yzc = dataSplit(ynode.text)
    pixelate(
        yzy,
        yzz,
        yzc,
        name=segnode.get("ID").zfill(6) + ".png",
        path="./pixelate/img/" + fn + "/yz/")

    c += 1
    prc= c*100/count
    if int(prc*10)>int((c-1)*1000/count):
        print "%.1f%%" % prc

print "Done"
