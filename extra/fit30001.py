#!/usr/bin/env python
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras.models import Sequential,Model
from keras.layers import Dense,Activation,Dropout,Input, BatchNormalization
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

import os
import pylab as py

from tools.tools import checkdir,load,save
from qcdlib import mellin

def get_data(k,channel,flavor,path2nptabs,original=False):
    data=np.load('%s/%s/%s/%s.npy'%(path2nptabs,k,channel,flavor))

    num_inputs=3
    num_outputs=2
    #--split into input (x) and output (y)
    x=data[:num_inputs]
    y=data[num_inputs:]

    x,y=x.T,y.T
    x_train_orig,x_test,y_train_orig,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
    xsc=MinMaxScaler()
    ysc=MinMaxScaler()
    x_train=xsc.fit_transform(x_train_orig)
    x_test=xsc.transform(x_test)
    y_train=ysc.fit_transform(y_train_orig)
    y_test=ysc.transform(y_test)

    if original: return x_train,x_test,y_train,y_test,x_train_orig,y_train_orig
    else: return x_train,x_test,y_train,y_test

def gen_model(EPOCHS, BATCH_SIZE, lr, k, channel, flavor, path2nptabs, path2nnmodels):

    print('\ngenerating model for %s, %s'%(channel,flavor))

    x_train,x_test,y_train,y_test,x_train_orig,y_train_orig = get_data(k,channel,flavor,path2nptabs,original=True)

    input=Input(shape= x_train[0].shape)
    x=Dense(120, activation='relu')(input)
    x = Dropout(0.01)(x)
    x=Dense(120, activation='relu')(x)
    x = Dropout(0.01)(x)
    x=Dense(120, activation='relu')(x)
    x = Dropout(0.01)(x)
    output=Dense(2)(x)
    model=Model(input,output)
    model.summary()
    model_optimizer=Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.00001) 
    model.compile(optimizer=model_optimizer,loss='mean_squared_error',metrics=['accuracy'])
    history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.2, verbose=1)

    checkdir('%s/%s/%s'%(path2nnmodels,k,channel))
    filename='%s/%s/%s/%s.json'%(path2nnmodels,k,channel,flavor)
    save({'model':model,'history':history,'x_train_orig':x_train_orig,'y_train_orig':y_train_orig},filename)

def plot_metrics(path2nnmodels,k,channel,flavor):
    checkdir('%s/%s/%s'%(path2nnmodels,k,channel))
    modeldata=load('%s/%s/%s/%s.json'%(path2nnmodels,k,channel,flavor))

    model,history=modeldata['model'],modeldata['history']

    nrows, ncols = 1, 2
    fig=plt.figure(figsize=(ncols*7,nrows*4))
    ax1=plt.subplot(nrows, ncols, 1)
    ax2=plt.subplot(nrows, ncols, 2)

    ax1.semilogy()

    ax1.plot(history.history['loss'])
    ax1.plot(history.history['val_loss'])
    ax1.set_ylabel('loss')
    ax1.set_xlabel('epoch')
    ax1.legend(['train','validation'], loc='upper left')

    ax2.plot(history.history['accuracy'])
    ax2.plot(history.history['val_accuracy'])
    ax2.set_ylabel('accuracy')
    ax2.set_xlabel('epoch')
    ax2.legend(['train','validation'], loc='upper left')

    plt.tight_layout()
    checkdir('gallery/%s/%s/%s'%(k,channel,flavor))
    filename='gallery/%s/%s/%s/metrics.png'%(k,channel,flavor)
    plt.savefig(filename)
    plt.close()
    print('Saving figure to %s'%filename)

def plot_predictions(path2nnmodels,k,channel,flavor,path2nptabs):
 
    x_train,x_test,y_train,y_test= get_data(k,channel,flavor,path2nptabs)

    y_test=y_test.T
    y_train=y_train.T

    #--load model data
    modeldata=load('%s/%s/%s/%s.json'%(path2nnmodels,k,channel,flavor))
    model = modeldata['model']

    y_pred=model.predict(x_test).T

    nrows, ncols = 1, 2
    fig=plt.figure(figsize=(ncols*7,nrows*4))
    ax1=plt.subplot(nrows, ncols, 1)
    ax2=plt.subplot(nrows, ncols, 2)

    ax1.scatter(y_test[0],y_pred[0],label='Predicted',s=0.5)
    ax1.plot(y_test[0], y_test[0], color='orange')
    ax1.legend()
    ax1.set_xlabel('Re($\sigma$)_test')
    ax1.set_ylabel('Re($\sigma$)_pred')

    ax2.scatter(y_test[1],y_pred[1],label='Predicted',s=0.5)
    ax2.plot(y_test[1], y_test[1], color='orange')
    ax2.legend()
    ax2.set_xlabel('Im($\sigma$)_test')
    ax2.set_ylabel('Im($\sigma$)_pred')
    checkdir('gallery/%s/%s/%s'%(k,channel,flavor))
    filename='gallery/%s/%s/%s/predictions.png'%(k,channel,flavor)
    plt.savefig(filename)
    plt.close()
    print('Saving figure to %s'%filename)


    y_pred=model.predict(x_train).T

    nrows, ncols = 1, 2
    fig=plt.figure(figsize=(ncols*7,nrows*4))
    ax1=plt.subplot(nrows, ncols, 1)
    ax2=plt.subplot(nrows, ncols, 2)

    ax1.scatter(y_train[0],y_pred[0],label='Predicted',s=0.5)
    ax1.plot(y_train[0], y_train[0], color='orange')
    ax1.legend()
    ax1.set_xlabel('Re($\sigma$)_train')
    ax1.set_ylabel('Re($\sigma$)_pred')

    ax2.scatter(y_train[1],y_pred[1],label='Predicted',s=0.5)
    ax2.plot(y_train[1], y_train[1], color='orange')
    ax2.legend()
    ax2.set_xlabel('Im($\sigma$)_train')
    ax2.set_ylabel('Im($\sigma$)_pred')
    checkdir('gallery/%s/%s/%s'%(k,channel,flavor))
    filename='gallery/%s/%s/%s/trainingset.png'%(k,channel,flavor)
    plt.savefig(filename)
    plt.close()
    print('Saving figure to %s'%filename)

if __name__=='__main__':
    path2nptabs='nptabs/'
    path2nnmodels='nnmodels/'

    #--generate the model for 30001
    k=30001
    channel='qA,qbB'
    for flav in ['ub','db','sb','cb','bb']:
        gen_model(10000,32,1e-4,k,channel,flav,path2nptabs,path2nnmodels)
        plot_metrics(path2nnmodels,k,channel,flav)
        plot_predictions(path2nnmodels,k,channel,flav,path2nptabs)
    channel='qbA,qB'
    for flav in ['u','d','s','c','b']:
        gen_model(10000,32,1e-4,k,channel,flav,path2nptabs,path2nnmodels)
        plot_metrics(path2nnmodels,k,channel,flav)
        plot_predictions(path2nnmodels,k,channel,flav,path2nptabs)
    channel='qA,gB'
    for flav in ['g']:
        gen_model(10000,32,1e-4,k,channel,flav,path2nptabs,path2nnmodels)
        plot_metrics(path2nnmodels,k,channel,flav)
        plot_predictions(path2nnmodels,k,channel,flav,path2nptabs)
    channel='gA,qB'
    for flav in ['u','d','s','c','b','ub','db','sb','cb','bb']:
        gen_model(10000,32,1e-4,k,channel,flav,path2nptabs,path2nnmodels)
        plot_metrics(path2nnmodels,k,channel,flav)
        plot_predictions(path2nnmodels,k,channel,flav,path2nptabs)


