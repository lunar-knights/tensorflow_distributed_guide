import os
import random
import shutil


abspath = os.path.split(os.path.realpath(__file__))[0]
# 数据文件夹
data_dir = abspath + "..\\data"
train_data_dir = abspath + "..\\traindata"
test_data_dir = abspath + "..\\testdata"

fpaths = []

for fname in os.listdir(data_dir):
    fpath = os.path.join(data_dir, fname)
    fpaths.append(fpath)
random.shuffle(fpaths)

fpaths_train = fpaths[0:3800]
fpaths_test = fpaths[3800:]

for fpath in fpaths_train:
    fs = os.path.split(fpath)
    shutil.copyfile(fpath,os.path.join(train_data_dir,fs[1]))

for fpath in fpaths_test:
    fs = os.path.split(fpath)
    shutil.copyfile(fpath,os.path.join(test_data_dir,fs[1]))

