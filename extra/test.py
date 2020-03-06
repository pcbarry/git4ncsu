#!/usr/bin/env python
import sys,os
import numpy as np
import pandas as pd

#--matplotlib
import matplotlib
matplotlib.use('Agg')
#matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
#matplotlib.rc('text',usetex=True)
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import pylab as py

import lhapdf

#--from tools
from tools.config import conf
from tools.tools import checkdir

#--from qcdlib
from qcdlib import aux,eweak,mellin,alphaS,pdfpion1

#--from dy
from obslib.dy import piontheoryml,fakepdf,reader

class TUNGSTEN():

    def __init__(self):
        col='EPPS16nlo_CT14nlo_W184'
        Set=0
        self.lha=lhapdf.mkPDF(col,Set)

    def get_pdfs(self,x,Q2):
      p=np.zeros(11)
      p[0]=self.lha.xfxQ2(21,x,Q2)
      p[1]=self.lha.xfxQ2( 2,x,Q2)
      p[2]=self.lha.xfxQ2(-2,x,Q2)
      p[3]=self.lha.xfxQ2( 1,x,Q2)
      p[4]=self.lha.xfxQ2(-1,x,Q2)
      p[5]=self.lha.xfxQ2( 3,x,Q2)
      p[6]=self.lha.xfxQ2(-3,x,Q2)
      p[7]=self.lha.xfxQ2( 4,x,Q2)
      p[8]=self.lha.xfxQ2(-4,x,Q2)
      p[9]=self.lha.xfxQ2( 5,x,Q2)
      p[10]=self.lha.xfxQ2(-5,x,Q2)
      p/=x
      return p



def get_xspace(k,part,flav=None):
    xspace=np.zeros(len(conf['dy-pion tabs'][k]['idx']))
    for i in range(len(xspace)):
        Q2   = conf['dy-pion tabs'][k]['Q2'][i]
        Jac  = conf['dy-pion tabs'][k]['Jac'][i]
        units= conf['dy-pion tabs'][k]['Units'][i]
        S    = conf['dy-pion tabs'][k]['S'][i]
        Y    = conf['dy-pion tabs'][k]['Y'][i]
        if part!='full': xspace[i]=conf['dy-pion'].get_xsec(Q2,S,Y,Q2,ilum='flavors',part=part,flav=flav) * Jac * units
        elif part=='full': xspace[i]=conf['dy-pion'].get_xsec(Q2,S,Y,Q2,ilum='normal',part=part,flav=flav) * Jac * units
    return xspace

def get_mellspace(k,part,flav=None):
    conf['dy-pion'].load_melltab_hybrid()
    mellspace=np.zeros(len(conf['dy-pion tabs'][k]['idx']))
    for i in range(len(mellspace)):
        Q2   = conf['dy-pion tabs'][k]['Q2'][i]
        Jac  = conf['dy-pion tabs'][k]['Jac'][i]
        units= conf['dy-pion tabs'][k]['Units'][i]
        mellspace[i]=conf['dy-pion'].get_xsec_mell_hybrid(k,i,Q2,part,flav=flav) * Jac * units
    return mellspace

def get_nnspace(k,part,flav=None):
    conf['dy-pion'].load_nn_hybrid()
    nnspace=np.zeros(len(conf['dy-pion tabs'][k]['idx']))
    for i in range(len(nnspace)):
        Q2   = conf['dy-pion tabs'][k]['Q2'][i]
        Jac  = conf['dy-pion tabs'][k]['Jac'][i]
        units= conf['dy-pion tabs'][k]['Units'][i]
        nnspace[i]=conf['dy-pion'].get_xsec_mell_hybrid_nn(k,i,Q2,part,flav=flav) * Jac * units
    return nnspace

def plot_mellin(mell,nn,k,part,flav,i):
    Z=conf['mellin'].Z
    nrows,ncols=1,2
    fig=py.figure(figsize=(4*ncols,4*nrows))

    ax=py.subplot(nrows,ncols,1)
    ax.plot(Z,mell.real,label='mellin')
    ax.plot(Z,nn.real,label='nn')
    ax.set_title(r'$x_F=%.3f$'%conf['dy-pion tabs'][k]['xF'][i],size=20)
    ax.set_xlabel(r'$Z$')
    ax.set_ylabel(r'Re[$\sigma$]')

    ax=py.subplot(nrows,ncols,2)
    ax.plot(Z,mell.imag,label='mellin')
    ax.plot(Z,nn.imag,label='nn')
    ax.set_title(r'$Q^2=%.3f$'%conf['dy-pion tabs'][k]['Q2'][i],size=20)
    ax.set_xlabel(r'$Z$')
    ax.set_ylabel(r'Im[$\sigma$]')

    ax.legend()

    py.tight_layout()
    if part=='full':
        checkdir('gallery/%s/%s'%(k,part))
        py.savefig('gallery/%s/%s/testmell%s.png'%(k,part,i))
        py.close()
    else:
        checkdir('gallery/%s/%s/%s'%(k,part,flav))
        py.savefig('gallery/%s/%s/%s/testmell%s.png'%(k,part,flav,i))
        py.close()

def plot_E615(tabs,x,mell,nn,k,part,flav):
    
    nrows,ncols=1,1
    fig=py.figure(figsize=(ncols*5,nrows*7))
    ax=py.subplot(nrows,ncols,1)

    Q2bins=[]
    Q2bins.append([20,23])
    Q2bins.append([23,28])
    Q2bins.append([28,35])
    Q2bins.append([35,40])
    Q2bins.append([40,45])
    Q2bins.append([45,50])
    Q2bins.append([50,60])
    Q2bins.append([60,65])
    Q2bins.append([65,75])
    
    tab={}
    tab['xF']=tabs['xF']
    tab['Q2']=tabs['Q2']
    tab['xspace']=x
    tab['mellspace']=mell
    tab['nnspace']=nn
    if 'g' in part:
        tab['xspace'],tab['mellspace'],tab['nnspace']=-x,-mell,-nn
    tab=pd.DataFrame(tab)
    for i in range(len(Q2bins)):
        d=tab.query('Q2>%f and Q2<%f'%(Q2bins[i][0],Q2bins[i][1]))
        ii=np.argsort(d['xF'].values)
        X=d['xF'].values[ii]
        xxsec=d['xspace'].values[ii]
        mellxsec=d['mellspace'].values[ii]
        nnxsec=d['nnspace'].values[ii]

        ax.plot(X[:-1],(xxsec*3**i)[:-1],color='#1f77b4',label='x-space')
        ax.plot(X[:-1],(mellxsec*3**i)[:-1],'*',color='#ff7f0e',label='using Mellin tables')
        ax.plot(X[:-1],(nnxsec*3**i)[:-1],color='#2ca02c',label='neural net')
        if i==0: ax.legend()

    ax.semilogy()
    ax.set_ylabel(r'$\frac{d\sigma}{dx_F d\sqrt{\tau}} *3**i$')
    if 'g' in part: ax.set_ylabel(r'$-\frac{d\sigma}{dx_F d\sqrt{\tau}} *3**i$')
    ax.set_xlabel(r'$x_F$')
    ax.set_title(r'dataset=%s'%k)

    py.tight_layout()
    if part=='full':
        checkdir('gallery/%s/%s'%(k,part))
        py.savefig('gallery/%s/%s/test.png'%(k,part))
        py.close()
    else:
        checkdir('gallery/%s/%s/%s'%(k,part,flav))
        py.savefig('gallery/%s/%s/%s/test.png'%(k,part,flav))
        py.close()

def plot_NA10(tabs,x,mell,nn,k,part,flav):

    nrows,ncols=1,1
    fig=py.figure(figsize=(ncols*5,nrows*7))
    ax=py.subplot(nrows,ncols,1)

    Q2bins=[]
    Q2bins.append([15,18])
    Q2bins.append([18,20])
    Q2bins.append([20,23])
    Q2bins.append([23,26])
    Q2bins.append([26,30])
    Q2bins.append([30,33])
    Q2bins.append([33,35])
    Q2bins.append([35,40.8])
    Q2bins.append([40.8,43])
    Q2bins.append([43,49])
    Q2bins.append([49,52])
    Q2bins.append([52,58])
    Q2bins.append([58,65])
    
    tab={}
    tab['xF']=tabs['xF']
    tab['Q2']=tabs['Q2']
    tab['xspace']=x
    tab['mellspace']=mell
    tab['nnspace']=nn
    if part=='qA,gB':
        tab['xspace'],tab['mellspace'],tab['nnspace']=-x,-mell,-nn
    tab=pd.DataFrame(tab)
    for i in range(len(Q2bins)):
        d=tab.query('Q2>%f and Q2<%f'%(Q2bins[i][0],Q2bins[i][1]))
        ii=np.argsort(d['xF'].values)
        X=d['xF'].values[ii]
        xxsec=d['xspace'].values[ii]
        mellxsec=d['mellspace'].values[ii]
        nnxsec=d['nnspace'].values[ii]

        ax.plot(X[:],(xxsec)[:],color='#1f77b4',label='x-space')
        ax.plot(X[:],(mellxsec)[:],'*',color='#ff7f0e',label='using Mellin tables')
        ax.plot(X[:],(nnxsec)[:],color='#2ca02c',label='neural net')
        if i==0: ax.legend()

    ax.semilogy()
    ax.set_ylabel(r'$\frac{d\sigma}{dx_F d\sqrt{\tau}}$')
    if part=='qA,gB': ax.set_ylabel(r'$-\frac{d\sigma}{dx_F d\sqrt{\tau}}$')
    ax.set_xlabel(r'$x_F$')
    ax.set_title(r'dataset=%s'%k)

    py.tight_layout()
    if part=='full':
        checkdir('gallery/%s/%s'%(k,part))
        py.savefig('gallery/%s/%s/test.png'%(k,part))
        py.close()
    else:
        checkdir('gallery/%s/%s/%s'%(k,part,flav))
        py.savefig('gallery/%s/%s/%s/test.png'%(k,part,flav))
        py.close()


if __name__=='__main__':

    conf['order']='NLO'
    conf['aux']=aux.AUX()
    conf['Q20']=conf['aux'].mc2
    conf['eweak']=eweak.EWEAK()
    conf['alphaS']=alphaS.ALPHAS()
    conf['mellin-pion']=mellin.MELLIN(npts=8,extended=True)
    conf['mellin']=conf['mellin-pion']

    conf['pdfA']=fakepdf.FAKEPDF()
    conf['pdf-pion']=conf['pdfA']
    conf['pdfB']=TUNGSTEN()


    conf['datasets']={}
    conf['datasets']['dy-pion']={}
    conf['datasets']['dy-pion']['filters']=[]
    conf['datasets']['dy-pion']['filters'].append("Q2>4.16**2")
    conf['datasets']['dy-pion']['filters'].append("Q2<8.34**2")
    conf['datasets']['dy-pion']['filters'].append("xF>0")
    conf['datasets']['dy-pion']['filters'].append("xF<0.9")
    conf['datasets']['dy-pion']['xlsx']={}
    conf['datasets']['dy-pion']['xlsx'][10001]='dy-pion/expdata/10001.xlsx'
    conf['datasets']['dy-pion']['xlsx'][10002]='dy-pion/expdata/10002.xlsx'
    conf['datasets']['dy-pion']['xlsx'][10003]='dy-pion/expdata/10003.xlsx'
    #conf['datasets']['dy-pion']['xlsx'][30001]='dy-pion/expdata/30001.xlsx'

    conf['dy-pion tabs']=reader.READER_PIONS().load_data_sets('dy-pion')
    conf['path2ml4jam']=os.environ['ML4JAM']

    conf['dy-pion']=piontheoryml.DY_PION()

    #conf['path2dytab-hybrid']='/Users/barryp/JAM/workspace/ml4jam/dy/melltabs/'
    conf['path2dytab-hybrid']='%s/grids/grids-dypion'%os.environ['FITPACK']
    conf['path2nntabs']='%s/dy/nnmodels/'%os.environ['ML4JAM']

    datasets=[10001,10002,10003]
    for k in datasets:
        parts=['qA,qbB','qbA,qB','qA,gB','gA,qB']
        for part in parts:
            #if part=='qA,qbB': flavs=['ub','db','sb','cb','bb']
            #if part=='qbA,qB': flavs=['u','d','s','c','b']
            if part=='qA,gB':  flavs=['g']
            #if part=='gA,qB':  flavs=['u','ub','d','db','s','sb','c','cb','b','bb']
            else: continue
            for f in flavs:
                xspacedat=get_xspace(k,part,flav=f)
                mellspacedat=get_mellspace(k,part,flav=f)
                nnspacedat=get_nnspace(k,part,flav=f)

                if k==10001: plot_E615(conf['dy-pion tabs'][k],xspacedat,mellspacedat,nnspacedat,k,part,f)
                else: plot_NA10(conf['dy-pion tabs'][k],xspacedat,mellspacedat,nnspacedat,k,part,f)
                for i in range(len(conf['dy-pion tabs'][k]['idx'])):
                    print('\nplotting the integrands')
                    mellinxsec=load('%s/dy/data/%s/%s/%s/%smell.dat'%(os.environ['ML4JAM'],k,part,f,i))
                    nnxsec=load('%s/dy/data/%s/%s/%s/%snn.dat'%(os.environ['ML4JAM'],k,part,f,i))

                    plot_mellin(mellinxsec,nnxsec,k,part,f,i)
        part='full'
        xspacedat=get_xspace(k,part)
        mellspacedat=get_mellspace(k,part)
        nnspacedat=get_nnspace(k,part)

        if k==10001: plot_E615(conf['dy-pion tabs'][k],xspacedat,mellspacedat,nnspacedat,k,part,None)
        else: plot_NA10(conf['dy-pion tabs'][k],xspacedat,mellspacedat,nnspacedat,k,part,None)

        for i in range(len(conf['dy-pion tabs'][k]['idx'])):
            print('\nplotting the integrands')
            mellinub=load('%s/dy/data/%s/%s/%sxsecmell.dat'%(os.environ['ML4JAM'],k,part,i))
            nnub=load('%s/dy/data/%s/%s/%sxsecnn.dat'%(os.environ['ML4JAM'],k,part,i))

            plot_mellin(mellinub,nnub,k,part,None,i)
