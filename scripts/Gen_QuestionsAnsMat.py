# -*- coding: utf-8 -*
home_dir = '../../../..'
import sys,os
sys.path.append(os.path.join(home_dir,'tools'))
from glob import glob
import numpy as np
import struct,os
from tqdm import tqdm
from tools import ReadFloatRawMat
'''
filename= './QuestionsAnsMat.dat'
QuesDir = './train+val'
'''
QuesDir = './test_linguisticfeas'
filename= './test_QuestionsAnsMat.dat'

QuesSet = os.path.join(home_dir,'edfiles/questions/questions.hed')
QuesNum=len(np.loadtxt(QuesSet,'str'))
print "Questions Number: %d" % QuesNum

with open(filename, "wb") as myfile:
    files = sorted(os.listdir(QuesDir))
    for file in tqdm(files):
        data=ReadFloatRawMat(os.path.join(QuesDir,file),QuesNum).flatten()
        myfile.write(struct.pack('<'+str(len(data))+'f',*data))