# -*- coding: utf-8 -*-
home_dir = '../../../..'
import sys,os
sys.path.append(os.path.join(home_dir,'tools'))
import mxnet as mx
from mxnet.io import DataIter,DataBatch
from tools import *
from mxnet.gluon import nn
from mxnet import nd,gluon,autograd,gpu,cpu
import numpy as np
from tqdm import tqdm
import os

class FileIter(DataIter):

    def __init__(self, batch_size=32,root_dir='../data',
                 data_dims=523,label_dims=43,
                 data_name="data", label_name="softmax_label",
                 last_batch_handle="pad"):
        super(DataIter, self).__init__()

        self.root_dir=root_dir
        self.batch_size = batch_size
        self.cursor = -batch_size
        self.last_batch_handle = last_batch_handle
        self.data_name = data_name
        self.label_name = label_name
        self.training_files = read_file_list(os.path.join(home_dir,'train_file.lst'))
        self.training_set_data = map(lambda i:os.path.join(self.root_dir,'linguistic_frameLevel_normalization',i+'.dat'),self.training_files)
        self.training_set_label = map(lambda i:os.path.join(self.root_dir,'DNNOutput_mcep_lf0_uv_normalization',i+'.dat'),self.training_files)
        self.num_trainingfiles = len(self.training_files)
        self.data_dims=data_dims
        self.label_dims=label_dims
        
    def __len__(self):
        return self.num_trainingfiles

    def read_data_files(self,filelst):
        data_lst=[]
        for i in filelst:
            data_lst.append(nd.array(ReadFloatRawMat(i,self.data_dims)))
        return data_lst
    
    def read_label_files(self,filelst):
        label_lst=[]
        for i in filelst:
            label_lst.append(nd.array(ReadFloatRawMat(i,self.label_dims)))
        return label_lst

    def reset(self):
        self.cursor = -self.batch_size

    def iter_next(self):
        self.cursor += self.batch_size
        return self.cursor < self.num_trainingfiles

    def next(self):
        if self.iter_next():
            return DataBatch(data=self.getdata(),
                             label=self.getlabel(),
                             pad=self.getpad())
        else:
            raise StopIteration

    def getdata(self):
        assert(self.cursor < self.num_trainingfiles), "DataIter needs reset."
        if self.cursor + self.batch_size <= self.num_trainingfiles:
            return self.read_data_files(self.training_set_data[self.cursor:self.cursor+self.batch_size])
        else:
            pad = self.batch_size - self.num_trainingfiles + self.cursor
            return self.read_data_files(self.training_set_data[self.cursor:]+self.training_set_data[:pad])

    def getlabel(self):
        assert(self.cursor < self.num_trainingfiles), "DataIter needs reset."
        if self.cursor + self.batch_size <= self.num_trainingfiles:
            return self.read_label_files(self.training_set_label[self.cursor:self.cursor+self.batch_size])
        else:
            pad = self.batch_size - self.num_trainingfiles + self.cursor
            return self.read_label_files(self.training_set_label[self.cursor:]+self.training_set_label[:pad])

    def getpad(self):
        if self.last_batch_handle == "pad" and self.cursor + self.batch_size > self.num_trainingfiles:
            return self.cursor + self.batch_size - self.num_trainingfiles
        else:
            return 0

class LSTM(nn.Block):
        def __init__(self, **kwargs):
            super(LSTM, self).__init__(**kwargs)
            self.encodeText=nn.Sequential()
            with self.name_scope():
                self.encodeText.add(gluon.rnn.LSTM(512, 2, bidirectional=True))
                self.encodeText.add(nn.Dense(127, flatten=False))

        def forward(self, text):
            return self.encodeText(text)

def L2LossMask(a,b,mask):
    #类似于gluon.loss.L2Loss(batch_axis=1)，但是可以用mask方式计算
    maskloss=[]
    maska = mx.nd.SequenceMask(a, mask, use_sequence_length=True)
    maskb = mx.nd.SequenceMask(b, mask, use_sequence_length=True)
    for i in range(a.shape[1]):
        index = int(mask[i].asscalar())
        maskloss.append(mx.nd.sum((maska[:index,i,:]-maskb[:index,i,:])**2)/(2*index*a.shape[2]))
    return mx.nd.concat(*maskloss, dim=0)

def train():
    epochs=30
    batch_size = 16
    data_dims=13
    label_dims=127
    root_dir='../data'
    
    #构造验证集数据------------------------------------------------------------------------------------------
    validation_files = read_file_list(os.path.join(home_dir,'val_file.lst'))
    validation_set_data_lst = map(lambda i:os.path.join(root_dir , 'linguistic_frameLevel_normalization', i+'.dat'), validation_files)
    validation_set_label_lst = map(lambda i:os.path.join(root_dir, 'DNNOutput_mcep_lf0_uv_normalization', i+'.dat'), validation_files)
    #构造验证集数据------------------------------------------------------------------------------------------
    
    net = LSTM()
    ctx = gpu()
    net.initialize(mx.initializer.Xavier(),ctx=ctx)
    dataiter = FileIter(batch_size=batch_size,root_dir=root_dir,data_dims=data_dims,label_dims=label_dims)
    square_loss1 = gluon.loss.L2Loss(batch_axis=1)
    trainer = gluon.Trainer(net.collect_params(), 'adam')
    num_examples = len(dataiter)
    Best_epoch = 0
    min_val_loss=float("inf")

    linguistic_dims = 913
    linguistic_Mat = ReadFloatRawMat('../data/QuestionsMat.dat', linguistic_dims)

    with open('loss_LSTM.txt','w') as loss_log:
        for epoch in xrange(epochs):
            total_loss = 0
            dataiter.reset()
            for index, batch in tqdm(enumerate(dataiter),total=num_examples/batch_size,unit="mini-batchs"):
                with autograd.record():
                    dur=map(lambda i:i.shape[0], batch.data)
                    data_mat=mx.nd.zeros((max(dur),batch_size,linguistic_dims+data_dims-1))
                    for data_index,data in enumerate(batch.data):
                        data_mat[:data.shape[0],data_index,:]=np.hstack((linguistic_Mat[data[:,0].asnumpy().astype(np.int)], data[:,1:].asnumpy()))

                    y = mx.nd.zeros((max(dur), batch_size, label_dims))
                    for label_index, label in enumerate(batch.label):
                        y[:label.shape[0],label_index,:]=label

                    y_hat = net(data_mat.as_in_context(ctx))
                    loss1 = L2LossMask(y_hat, y.as_in_context(ctx), nd.array(dur,ctx=ctx))
                    loss1.backward()
                trainer.step(batch_size)
                loss=nd.mean(loss1).asscalar()
                loss_log.write(str(loss)+'\n')
                loss_log.flush()
                total_loss += loss
                if index%50 == 0:
                    tqdm.write("Epoch %d, Batch: %d/%d, average loss in this batch: %f Best epoch: %d"%(epoch,index, num_examples/batch_size, loss, Best_epoch))
            
            print("Epoch %d finished, average loss of training set: %f" % (epoch, total_loss/(num_examples/batch_size)))
            val_loss=[]
            for i in tqdm(xrange(len(validation_files))):
                validation_set_data = ReadFloatRawMat(validation_set_data_lst[i],data_dims)
                validation_set_data = np.hstack((linguistic_Mat[validation_set_data[:,0].astype(np.int)],validation_set_data[:,1:]))
                validation_set_data = nd.array(validation_set_data).expand_dims(axis=1)

                validation_set_label = nd.array(ReadFloatRawMat(validation_set_label_lst[i],label_dims))
                validation_set_label = nd.array(validation_set_label).expand_dims(axis=1)

                validation_set_y_hat = net(validation_set_data.as_in_context(ctx))
                loss1 = square_loss1(validation_set_y_hat, validation_set_label.as_in_context(ctx)).asscalar()
                val_loss.append(loss1)
            val_loss=np.mean(val_loss)
            print("\n---loss of validation set:  %f---\n" % val_loss)
            if(val_loss < min_val_loss):
                min_val_loss = val_loss
                Best_epoch = epoch
                net.save_parameters('LSTM.parms')
                print("---validation set got a smaller loss---\n---------------Save net----------------\n")
            
        print("Best validation set loss is %f, in epoch %d\n"%(min_val_loss,Best_epoch))

def test():
    net = LSTM()
    ctx = gpu()
    net.load_parameters('LSTM.parms',ctx=ctx)

    test_inDir = '../data/test_linguistic_frameLevel'
    train_val_inDir = '../data/linguistic_frameLevel_normalization'
    outDir = '../data/Predicted_BLSTM_acousticfeas'
    mean_in_file = '../data/MeanStd_BLSTM_data.mean'
    mean_out_file = '../data/MeanStd_BLSTM_label.mean'
    test_linguisticAnsMat = '../data/test_QuestionsAnsMat'
    SpectrumDir = outDir+os.sep+'SPE'
    F0Dir = outDir+os.sep+'LF0'
    UVDir = outDir+os.sep+'UV'
    SaveMkdir(outDir)
    SaveMkdir(SpectrumDir)
    SaveMkdir(F0Dir)
    SaveMkdir(UVDir)
    
    mean_in = ReadFloatRawMat(mean_in_file,13)
    mean_out = ReadFloatRawMat(mean_out_file, 127)

    linguistic_dims = 913
    linguisticAnsMat = ReadFloatRawMat(os.path.join(home_dir, 'UnitVec_ConSyn/Unit2Vec_64_epochs50/Training_data/data/QuestionsMat.dat'), linguistic_dims)
    test_linguisticAnsMat = ReadFloatRawMat('../data/test_QuestionsAnsMat.dat', linguistic_dims)

    #lst file
    train_val_test_lst = read_file_list(os.path.join(home_dir,'train+val+test_file.lst'))
    train_val_lst = read_file_list(os.path.join(home_dir,'train+val_file.lst'))
    test_lst = read_file_list(os.path.join(home_dir,'test_file.lst'))


    SPEDim = 41

    for file in tqdm(train_val_test_lst):
        SpectrumFile = SpectrumDir+os.sep+file+'.spe'
        F0File = F0Dir+os.sep+file+'.lf0'
        UVFile = UVDir+os.sep+file+'.uv' 

        if file in train_val_lst:
            inFile = train_val_inDir+os.sep+file+'.dat'  
            inData = ReadFloatRawMat(inFile,13)
            inData = mx.nd.array(np.hstack((linguisticAnsMat[inData[:,0].astype(np.int)],inData[:,1:])),ctx=ctx)
        else:
            inFile = test_inDir+os.sep+file+'.dat' 
            inData = ReadFloatRawMat(inFile,13)
            inData[:,1:] = (inData[:,1:]-mean_in[0,1:])/mean_in[1,1:]
            inData = mx.nd.array(np.hstack((test_linguisticAnsMat[inData[:,0].astype(np.int)],inData[:,1:])),ctx=ctx)
        outData = net(inData.expand_dims(axis=1)).flatten().asnumpy()
        outData = outData*mean_out[1]+mean_out[0]
        
        SpectrumData = outData[:,0:SPEDim*3]
        F0Data = outData[:,SPEDim*3:SPEDim*3+3]
        UVData = (outData[:,-1]>0.5).astype(np.int)
        
        WriteArrayFloat(SpectrumFile,SpectrumData)
        WriteArrayFloat(F0File,F0Data)
        WriteArrayFloat(UVFile,UVData)  

if __name__=='__main__':
    train()
    #test()
