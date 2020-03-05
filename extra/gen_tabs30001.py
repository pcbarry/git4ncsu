#!/usr/bin/env python
import os,sys
import numpy as np

import lhapdf

#--from tools
from tools.tools import load,save,checkdir
from tools.config import load_config,conf
#--from qcdlib
from qcdlib import mellin,aux,alphaS,eweak
#-- from dy
import obslib.dy.piontheory
import obslib.dy.reader
import obslib.dy.fakepdf as fakepdf

def gen_melltab():
    conf['path2dytab-hybrid']='melltabs/'

    conf['dy-pion']=obslib.dy.piontheory.DY_PION()
    conf['dy-pion'].mellin=conf['mellin']
    conf['dy-pion'].gen_melltab_hybrid()

def save_as_np(channel,flavor,path2melltabs,path2nptabs):

    print 'saving %s, %s as nptabs'%(channel,flavor)

    datasets=[30001]
    melltabs={}
    idx={}
    for k in datasets:
        melltabs[k]={}
        filenames=os.listdir('%s/%s'%(path2melltabs,k))
        idx[k]=[]
        for f in filenames:
            idx[k].append(int(f.split('.')[0]))
            melltabs[k][idx[k][-1]]=load('%s/%s/%s'%(path2melltabs,k,f))

    mellinN=conf['mellin'].N
    realN=mellinN.real
    imagN=mellinN.imag


    length=0
    for k in datasets:
        for i in idx[k]:
            if i not in conf['dy-pion tabs'][k]['idx']: continue
            for c in melltabs[k][i]:
                if c==channel:
                    for f in melltabs[k][i][c]:
                        if f==flavor:
                            for j in range(len(melltabs[k][i][c][f])):
                                length+=1

    inputs={}
    inputs['xF']=np.zeros(length)
    inputs['ReN']=np.zeros(length)
    inputs['ImN']=np.zeros(length)

    outputs={}
    outputs['Re sig']=np.zeros(length)
    outputs['Im sig']=np.zeros(length)

    cnt=0
    for k in datasets:
        for i in idx[k]:
            for j in range(len(conf['dy-pion tabs'][k]['idx'])):
                if i==conf['dy-pion tabs'][k]['idx'][j]:
                    xF=conf['dy-pion tabs'][k]['xF'][j]
            for chan in melltabs[k][i]:
                if chan==channel:
                    for flav in melltabs[k][i][chan]:
                        if flav==flavor:
                            for j in range(len(melltabs[k][i][chan][flav])):
                                #--append values into inputs,outputs
                                inputs['xF'][cnt]=xF
                                inputs['ReN'][cnt]=realN[j]
                                inputs['ImN'][cnt]=imagN[j]
                                
                                outputs['Re sig'][cnt]=melltabs[k][i][chan][flav][j].real
                                outputs['Im sig'][cnt]=melltabs[k][i][chan][flav][j].imag
                                cnt+=1

    numparams=len(inputs)+len(outputs)
    par=['xF','ReN','ImN','Re sig','Im sig']
    nptabs=np.zeros(shape=(numparams,length))
    for i in range(numparams):
        if i<3: nptabs[i]=inputs[par[i]]
        else: nptabs[i]=outputs[par[i]]

    checkdir(path2nptabs)
    checkdir('%s/%s/%s'%(path2nptabs,datasets[0],channel))
    np.save('%s/%s/%s/%s'%(path2nptabs,datasets[0],channel,flavor),nptabs)

if __name__=='__main__':

    conf['datasets']={}
    conf['datasets']['dy-pion']={}
    conf['datasets']['dy-pion']['filters']=[]
    conf['datasets']['dy-pion']['filters'].append("Q2>4.16**2")
    conf['datasets']['dy-pion']['filters'].append("Q2<8.34**2")
    conf['datasets']['dy-pion']['filters'].append("xF>0") 
    conf['datasets']['dy-pion']['filters'].append("xF<0.9") 
    conf['datasets']['dy-pion']['xlsx']={}
    conf['datasets']['dy-pion']['xlsx'][30001]='30001.xlsx'
    
    conf['aux']=aux.AUX()
    conf['Q20']=conf['aux'].mc2
    conf['eweak']=eweak.EWEAK()
    conf['order']='NLO'
    conf['alphaS']=alphaS.ALPHAS()
    conf['mellin']=mellin.MELLIN()
    conf['pdfB']=fakepdf.FAKEPDF()

    conf['dy-pion tabs']=obslib.dy.reader.READER_PIONS().load_data_sets('dy-pion')

    gen_melltab() #--only turn on to change the mellin tabs; already calculated

    path2melltabs='melltabs/'
    path2nptabs  ='nptabs/'
    #--save for dy-pion
    for flav in ['ub','db','sb','cb','bb']:
        save_as_np('qA,qbB',flav,path2melltabs,path2nptabs)
    for flav in ['u','d','s','c','b']:
        save_as_np('qbA,qB',flav,path2melltabs,path2nptabs)
    for flav in ['g']:
        save_as_np('qA,gB',flav,path2melltabs,path2nptabs)
    for flav in ['u','d','s','c','b','ub','db','sb','cb','bb']:
        save_as_np('gA,qB',flav,path2melltabs,path2nptabs)


