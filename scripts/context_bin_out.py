home_dir = '../../../..'
import sys,os
sys.path.append(os.path.join(home_dir,'tools'))
import re
import numpy as np
from tqdm import tqdm
from tools import *

def Get_Ques2Regular(ques_file):
    lines = filter(lambda i:i.strip()!='', open(ques_file,'rt').readlines())
    ques_lists = []
    for line in lines:
        ques_list = []
        sub_ques = re.findall(r'[{](.*)[}]', line)[0].split(',')
        for q in sub_ques:
            q = q.replace('*',r'.*').replace('?',r'.').replace('$',r'\$')\
                 .replace('+',r'\+').replace('|',r'\|').replace('^',r'\^')
            q = re.sub(r'^([a-z])',r'^\1', q)
            # Compile pattern is Very important. Make 10X faster than originals!
            # Original(1) ques_list.append(q)
            ques_list.append(re.compile(q))
        ques_lists.append(ques_list)
    return ques_lists

def fulllab2ling(str, ques_lists):
    linguistic_vec = np.zeros(len(ques_lists), dtype=np.float32)
    for i, sub_ques in enumerate(ques_lists):
        for sub_que in sub_ques:
            # Original(2) re.match(sub_que, q)
            if(sub_que.match(lab)):
                linguistic_vec[i] = 1
                break
    return linguistic_vec

if __name__ == '__main__':
    labdir   = os.path.join(home_dir,'labels/testlab')
    que_file = os.path.join(home_dir,'edfiles/questions/questions.hed')
    ref_list = np.loadtxt(os.path.join(home_dir,'test_file.lst'),'str')

    outdir   = "./test_linguisticfeas"
    SaveMkdir(outdir)
    ques_lists = Get_Ques2Regular(que_file)
    ref_files = glob(os.path.join(labdir,'*.lab'))
    for name in tqdm(sorted(ref_files)):
        basename = os.path.basename(name)[:-4]
        if basename in  ref_list:
            tqdm.write('process %s' % basename)
            lab_file = os.path.join(labdir, basename+'.lab')
            linguistic_file = os.path.join(outdir, basename+'.dat')

            labs = open(lab_file,'rt').readlines()
            linguistic_Mat = np.zeros((len(labs), len(ques_lists)), dtype=np.float32)
            for i,lab in enumerate(labs):
                linguistic_Mat[i] = fulllab2ling(lab, ques_lists)
            linguistic_Mat.tofile(linguistic_file)