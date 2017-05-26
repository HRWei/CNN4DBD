import os

data_dir="./pixelate/img"
xz_dir=data_dir+"/BambooKerasDataV2/xz/"
path=os.listdir(xz_dir)

for i in path:
    if len(i)<10:
        print i
        os.remove(xz_dir+i)

print len(os.listdir(xz_dir))
print "Done!"
