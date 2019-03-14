# -*- coding: utf-8 -*
home_dir = '../../../..'
import sys,os
sys.path.append(os.path.join(home_dir,'tools'))
from tools import *
from functions import *
import os
import numpy as np
from collections import namedtuple

def cutedlab2DNNInput(cutedlabDir,outDir):
    SaveMkdir(outDir)
    k=0
    for item in sorted(os.listdir(cutedlabDir)):
        print 'process %s' % item
        cutedLabFile = cutedlabDir+os.sep+item
        
        outFile = outDir+os.sep+item.split('.')[0]+'.dat'
        fout = open(outFile,'wb')

        lines = open(cutedLabFile,'rt').read().splitlines() 
        for i in range(len(lines)):
                if lines[i]=='':continue
                if lines[i].split()[2].endswith('[2]'):
                    phone_start = int(round(float(lines[i].split()[0])/50000.0))
                    phone_end = int(round(float(lines[i+4].split()[1])/50000.0))
                state_start = int(round(float(lines[i].split()[0])/50000.0))
                state_end   = int(round(float(lines[i].split()[1])/50000.0))
                
                for frame in range(state_start,state_end):
                    fout.write(struct.pack('<1f',*[k]))
                    ####feas[:5]是该帧所在的状态编号###############
                    feas=np.zeros(12) 
                    if lines[i].split()[2].endswith('[2]'):
                        lis=[1,0,0,0,0]
                        feas[:5]=lis
                    if lines[i].split()[2].endswith('[3]'):
                        lis=[0,1,0,0,0]
                        feas[:5]=lis
                    if lines[i].split()[2].endswith('[4]'):
                        lis=[0,0,1,0,0]
                        feas[:5]=lis
                    if lines[i].split()[2].endswith('[5]'):
                        lis=[0,0,0,1,0]
                        feas[:5]=lis
                    if lines[i].split()[2].endswith('[6]'):
                        lis=[0,0,0,0,1]
                        feas[:5]=lis
                    ################################################
                    feas[5]=state_end-state_start #该帧所在状态的时长
                    feas[6]=phone_end-phone_start #该帧所在音素的时长
                    ####feas[7]是该帧在当前状态的前向位置###########
                    try:
                        feas[7]=(frame-state_start)/float(state_end-1-state_start)
                    except:
                        feas[7]=0.5
                    feas[8]=1-feas[7] #当前状态的后向位置
                    #################################################
                    ####feas[8]是该帧在当前音素的前向位置############
                    try:
                        feas[9]=(frame-phone_start)/float(phone_end-1-phone_start)
                    except:
                        feas[9]=0.5        
                    feas[10]=1-feas[9]#当前音素的后向位置
                    feas[11]=(state_end-state_start)/float(phone_end-phone_start)#该帧所在的状态的时长在当前音素的时长中的比重
                    fout.write(struct.pack('<12f',*feas))        
                    if frame==phone_end-1:
                        k+=1
        fout.close()

if __name__=='__main__':
    lab2Lingistic = 0
    acoustic2DNNOut = 0
    norm = 1
    syn_cal_distortion = 0
    
    if lab2Lingistic:
        ################# DNN input preparation
        cutedlabDir = os.path.join(home_dir,'labels/cutedlab')
        DNNInputDir = '../data/linguistic_frameLevel'
        cutedlab2DNNInput(cutedlabDir,DNNInputDir)
        
        # test input preparation
        durlabDir = os.path.join(home_dir,'gen/yanping/sd')
        DNNInputDir = '../data/test_linguistic_frameLevel'
        ref_file = os.path.join(home_dir,'test_file.lst')
        durDir_2_cutedlabDir(durlabDir,cutedlabDir,ref_file=ref_file)
        cutedlab2DNNInput(cutedlabDir,DNNInputDir)
    
    if acoustic2DNNOut:
        ############# DNN output preparation        
        f0Dir  = os.path.join(home_dir,'audio/F0')
        speDir = os.path.join(home_dir,'audio/dfmcep40')
        DNNoutDir = '../data/DNNOutput_mcep_lf0_uv'
        speDim = 41
        F0Spe2DNNOutput(f0Dir,speDir,speDim,DNNoutDir)
                
    if norm:
        DNNInputDir = '../data/linguistic_frameLevel'
        DNNoutDir   = '../data/DNNOutput_mcep_lf0_uv'
        ############## normalization ##############
        #同时输出均值文件
        Normalization_MeanStd_Dir(DNNInputDir,13, range(1,13), None,'BLSTM','data', ref_file = os.path.join(home_dir,'train_file.lst'))
        #归一化验证集
        mean_file_data = ReadFloatRawMat('../data/MeanStd_BLSTM_data.mean',13)
        Normalization_MeanStd_Dir(DNNInputDir,13, range(1,13), mean_file_data,'BLSTM','data', ref_file = os.path.join(home_dir,'val_file.lst'))

        #同时输出均值文件
        Normalization_MeanStd_Dir(DNNoutDir,  127, range(127),None,'BLSTM','label',ref_file = os.path.join(home_dir,'train_file.lst'))
        #归一化验证集
        mean_file_label = ReadFloatRawMat('../data/MeanStd_BLSTM_label.mean',127)
        Normalization_MeanStd_Dir(DNNoutDir,  127, range(127),mean_file_label,'BLSTM','label',ref_file = os.path.join(home_dir,'val_file.lst'))
    
    #Train model then goto test part

    if syn_cal_distortion:
        outDir = '../data/Predicted_BLSTM_acousticfeas'
        
        #### MLPG, model output to f0 and spectral feature sequence
        MeanStd_file = '../data/MeanStd_BLSTM_label.mean'
        
        # Generate test wav Linux
        list_file_name = read_file_list(os.path.join(home_dir,'test_file.lst'))
        #DNN_MLPG(outDir, 127, MeanStd_file, list_file_name=list_file_name,cal_GV = False, syn=True, compute_distortion=False)
        
        # Calculate reconstruction error Linux MLPG
        list_file_name = read_file_list(os.path.join(home_dir,'train+val_file.lst'))
        #DNN_MLPG(outDir, 127, MeanStd_file, list_file_name=list_file_name, MLPG=True, cal_GV = False, syn=False, compute_distortion=False)
        
        # Calculate reconstruction error Windows
        ref_cep_dir = r'\\172.16.46.80\xzhou2\Yanping13k_IFLYFE\audio\dfmcep40'
        ref_f0_dir = r'\\172.16.46.80\xzhou2\Yanping13k_IFLYFE\audio\F0'
        DNN_MLPG(outDir, 127, MeanStd_file, list_file_name=list_file_name,  MLPG=False, cal_GV = False, syn=False, compute_distortion=True,ref_cep_dir=ref_cep_dir, ref_f0_dir = ref_f0_dir)
