#!/usr/bin/env python
from scipy.integrate import fixed_quad
from scipy.special import gamma
import numpy as np
from tools.tools import load, save, checkdir
from tools.config import conf
from sklearn.preprocessing import MinMaxScaler

class DY_PION:

  def __init__(self):
    self.aux=conf['aux']
    if conf['order']=='LO':  self.iord=0
    if conf['order']=='NLO': self.iord=1
    #self.mellin=conf['mellin']
    if 'mellin-pion' in conf: self.mellin_ext=conf['mellin-pion']
    self.eweak=conf['eweak']
    self.CF=self.aux.CF
    self.TF=self.aux.TF
    self.eU2=4./9.0
    self.eD2=1./9.0

  def beta(self,a,b):
    return gamma(a)*gamma(b)/gamma(a+b)

  def quad(self,f,xmin,xmax):
    #return _quad(f,xmin,xmax)[0]
    return fixed_quad(f,xmin,xmax,n=5)[0]

  def get_x1(self,z,y):
    out=self.tau/z*(1-(1-y)*(1-z))/(1-y*(1-z))
    return out**0.5*np.exp(self.Y)
 
  def get_x2(self,z,y):
    out=self.tau/z*(1-y*(1-z))/(1-(1-y)*(1-z))
    return out**0.5*np.exp(-self.Y)

  def _get_lum(self,fA,fB,channel): 

   if channel=='qA,qbB':

     out= self.eU2*(fA[1]*fB[2])\
         +self.eD2*(fA[3]*fB[4])\
         +self.eD2*(fA[5]*fB[6])
     if self.Nf==4: out+=self.eU2*(fA[7]*fB[8])
     if self.Nf==5: out+=self.eU2*(fA[7]*fB[8])+self.eD2*(fA[9]*fB[10])

   elif channel=='qbA,qB':

     out= self.eU2*(fB[1]*fA[2])\
         +self.eD2*(fB[3]*fA[4])\
         +self.eD2*(fB[5]*fA[6])
     if self.Nf==4: out+=self.eU2*(fB[7]*fA[8])
     if self.Nf==5: out+=self.eU2*(fB[7]*fA[8])+self.eD2*(fB[9]*fA[10])

   elif channel=='qA,gB':

     out= self.eU2*(fA[1]+fA[2])*fB[0]\
         +self.eD2*(fA[3]+fA[4])*fB[0]\
         +self.eD2*(fA[5]+fA[6])*fB[0]
     if self.Nf==4: out+=self.eU2*(fA[7]+fA[8])*fB[0] 
     if self.Nf==5: out+=self.eU2*(fA[7]+fA[8])*fB[0]+self.eD2*(fA[9]+fA[10])*fB[0]

   elif channel=='gA,qB':

     out= self.eU2*(fB[1]+fB[2])*fA[0]\
         +self.eD2*(fB[3]+fB[4])*fA[0]\
         +self.eD2*(fB[5]+fB[6])*fA[0]
     if self.Nf==4: out+=self.eU2*(fB[7]+fB[8])*fA[0]
     if self.Nf==5: out+=self.eU2*(fB[7]+fB[8])*fA[0]+self.eD2*(fB[9]+fB[10])*fA[0]

   return out 

  def get_flavors(self,fA,fB,channel): 

   if channel=='qA,qbB':

     if self.flav=='ub': out=self.eU2*(fA[1]*fB[2])
     if self.flav=='db': out=self.eD2*(fA[3]*fB[4])
     if self.flav=='sb': out=self.eD2*(fA[5]*fB[6])
     if self.flav=='cb': out=self.eU2*(fA[7]*fB[8])
     if self.flav=='bb': out=self.eD2*(fA[9]*fB[10])

   elif channel=='qbA,qB':

     if self.flav=='u': out=self.eU2*(fB[1]*fA[2])
     if self.flav=='d': out=self.eD2*(fB[3]*fA[4])
     if self.flav=='s': out=self.eD2*(fB[5]*fA[6])
     if self.flav=='c': out=self.eU2*(fB[7]*fA[8])
     if self.flav=='b': out=self.eD2*(fB[9]*fA[10])

   elif channel=='qA,gB':

     out= self.eU2*(fA[1]+fA[2])*fB[0]\
         +self.eD2*(fA[3]+fA[4])*fB[0]\
         +self.eD2*(fA[5]+fA[6])*fB[0]
     if self.Nf==4: out+=self.eU2*(fA[7]+fA[8])*fB[0] 
     if self.Nf==5: out+=self.eU2*(fA[7]+fA[8])*fB[0]+self.eD2*(fA[9]+fA[10])*fB[0]

   elif channel=='gA,qB':

     if self.flav=='u':  out=self.eU2*fB[1]*fA[0]
     if self.flav=='ub': out=self.eU2*fB[2]*fA[0]
     if self.flav=='d':  out=self.eD2*fB[3]*fA[0]
     if self.flav=='db': out=self.eD2*fB[4]*fA[0]
     if self.flav=='s':  out=self.eD2*fB[5]*fA[0]
     if self.flav=='sb': out=self.eD2*fB[6]*fA[0]
     if self.flav=='c':  out=self.eU2*fB[7]*fA[0]
     if self.flav=='cb': out=self.eU2*fB[8]*fA[0]
     if self.flav=='b':  out=self.eD2*fB[9]*fA[0]
     if self.flav=='bb': out=self.eD2*fB[10]*fA[0]

   return out 

  def _get_lum_flav(self,fA,fB,flav): 

    if   flav=='g' : return fA*fB[0]
    elif flav=='u' : return fA*fB[1]
    elif flav=='ub': return fA*fB[2]
    elif flav=='d' : return fA*fB[3]
    elif flav=='db': return fA*fB[4]
    elif flav=='s' : return fA*fB[5]
    elif flav=='sb': return fA*fB[6]
    elif flav=='c' : return fA*fB[7]
    elif flav=='cb': return fA*fB[8]
    elif flav=='b' : return fA*fB[9]
    elif flav=='bb': return fA*fB[10]

  def get_lum(self,z,y,channel): 
    x1=self.get_x1(z,y)
    x2=self.get_x2(z,y)
    if x1>1 or x2>1: return 0  

    if self.ilum=='normal':
      fA=conf['pdfA'].get_pdfs(x1,self.muF2)
      fB=conf['pdfB'].get_pdfs(x2,self.muF2)
      return self._get_lum(fA,fB,channel)

    elif self.ilum=='mell-real':

      return np.real(x1**(-self.n) * x2**(-self.m))

    elif self.ilum=='mell-imag':

      return np.imag(x1**(-self.n) * x2**(-self.m))

    elif self.ilum=='hybrid-real':

      fA=np.real(x1**(-self.n))
      fB=conf['pdfB'].get_pdfs(x2,self.muF2)
      return self._get_lum_flav(fA,fB,self.flav)

    elif self.ilum=='hybrid-imag':

      fA=np.imag(x1**(-self.n))
      fB=conf['pdfB'].get_pdfs(x2,self.muF2)
      return self._get_lum_flav(fA,fB,self.flav)

    if self.ilum=='flavors':
      fA=conf['pdfA'].get_pdfs(x1,self.muF2)
      fB=conf['pdfB'].get_pdfs(x2,self.muF2)
      return self.get_flavors(fA,fB,channel)

  def get_jac(self,z,y):
    return 1.0/(1-y*(1-z))/(1-(1-y)*(1-z))

  #--qqb channel

  #--part 0

  def Cqqb0(self,f):
    return (1+self.iord*self.CF*self.aS/np.pi*(3/2.*np.log(self.Q2/self.muF2)+2*np.pi**2/3-6))*0.5*(f(0)+f(1))
  
  def xsecCqqb0AB(self):
    return  self.Cqqb0(lambda y: self.get_lum(1,y,'qA,qbB')*self.get_jac(1,y))

  def xsecCqqb0BA(self):
    return  self.Cqqb0(lambda y: self.get_lum(1,y,'qbA,qB')*self.get_jac(1,y))

  #--part 1

  def Cqqb1(self,z,f):
    plus =np.log(self.Q2/self.muF2)*(1/(1-z)*((1+z*z)*f(z)-2*f(1)) + 2*f(1)*np.log(1-self.tau)/(1-self.tau))
    plus+=2*np.log(1-z)/(1-z)*((1+z*z)*f(z)-2*f(1)) + 2*f(1)*np.log(1-self.tau)**2/(1-self.tau)
    plus+=-1/(1-z)*(1+z*z)*f(z)*np.log(z) + 2*f(1)*(1-np.pi**2/6)/(1-self.tau)
    return 0.5*self.iord*self.CF*self.aS/np.pi*(plus+(1-z)*f(z))

  def dxsecCqqb1AB(self,z):
    return  self.Cqqb1(z,lambda z:self.get_lum(z,0,'qA,qbB')*self.get_jac(z,0)+self.get_lum(z,1,'qA,qbB')*self.get_jac(z,1))

  def dxsecCqqb1BA(self,z):
    return  self.Cqqb1(z,lambda z:self.get_lum(z,0,'qbA,qB')*self.get_jac(z,0)+self.get_lum(z,1,'qbA,qB')*self.get_jac(z,1))

  def xsecCqqb1AB(self):
    return self.quad(np.vectorize(self.dxsecCqqb1AB),self.tau,1)

  def xsecCqqb1BA(self):
    return self.quad(np.vectorize(self.dxsecCqqb1BA),self.tau,1)

  #--part 2

  def aux1(self,z,y):
    return 0.5*(1+(1-z)**2/z*y*(1-y))

  def Cqqb2(self,z,y,f):
    plus10=1.0/y*(f(z,y)*self.aux1(z,y)-f(z,0)*self.aux1(z,0))
    plus00=1.0/y*(f(1,y)*self.aux1(1,y)-f(1,0)*self.aux1(1,0))
    plus11=1.0/(1-y)*(f(z,y)*self.aux1(z,y)-f(z,1)*self.aux1(z,1))
    plus01=1.0/(1-y)*(f(1,y)*self.aux1(1,y)-f(1,1)*self.aux1(1,1))
    return self.iord*self.CF*self.aS/np.pi\
          *(  1/(1-z)*((1+z*z)*(plus10+plus11)-2*(plus00+plus01))\
             +2*np.log(1-self.tau)*(plus00+plus01)/(1-self.tau)\
             -2*(1-z)*f(z,y)* self.aux1(z,y))

  def dxsecCqqb2AB(self,z,y):
    return self.Cqqb2(z,y,lambda z,y: self.get_lum(z,y,'qA,qbB')*self.get_jac(z,y))

  def dxsecCqqb2BA(self,z,y):
    return self.Cqqb2(z,y,lambda z,y: self.get_lum(z,y,'qbA,qB')*self.get_jac(z,y))

  def xsecCqqb2AB(self):
    func=lambda z: self.quad(np.vectorize(lambda y: self.dxsecCqqb2AB(z,y)),0,1)
    return self.quad(np.vectorize(func),self.tau,1)

  def xsecCqqb2BA(self):
    func=lambda z: self.quad(np.vectorize(lambda y: self.dxsecCqqb2BA(z,y)),0,1)
    return self.quad(np.vectorize(func),self.tau,1)

  #--combined

  def xsecCqqbAB(self):
    return self.xsecCqqb0AB()+self.xsecCqqb1AB()+self.xsecCqqb2AB()

  def xsecCqqbBA(self):
    return self.xsecCqqb0BA()+self.xsecCqqb1BA()+self.xsecCqqb2BA()

  def xsecCqqb(self):
    return self.xsecCqqbAB() + self.xsecCqqbBA()

  #--qg channel

  #--part 1

  def Cqg1(self,z): 
    return self.iord*self.TF*self.aS/2/np.pi\
          *( (z**2+(1-z)**2) *(np.log(self.Q2/self.muF2)+np.log((1-z)**2/z))+2*z*(1-z))

  def dxsecCqg1AB(self ,z): 
    return  self.Cqg1(z)*self.get_lum(z,0,'qA,gB')*self.get_jac(z,0)

  def xsecCqg1AB(self):
    return self.quad(np.vectorize(self.dxsecCqg1AB),self.tau,1)

  def dxsecCqg1BA(self ,z): 
    return  self.Cqg1(z)*self.get_lum(z,1,'gA,qB')*self.get_jac(z,1)

  def xsecCqg1BA(self):
    return self.quad(np.vectorize(self.dxsecCqg1BA),self.tau,1)

  #--part 2

  def aux2(self,z,y):
    return 1+(1-z)**2/z*y*(1-y)

  def Cqg2AB(self,z,y,f):
    plus=1.0/y*(f(z,y)*self.aux2(z,y)-f(z,0)*self.aux2(z,0))
    return self.iord*self.TF*self.aS/2/np.pi\
          *( (z*z+(1-z)**2)*plus+(2*z*(1-z)+(1-z)**2*y)*f(z,y)*self.aux2(z,y))

  def Cqg2BA(self,z,y,f):
    plus=1.0/(1-y)*(f(z,y)*self.aux2(z,1-y)-f(z,1)*self.aux2(z,0))
    return self.iord*self.TF*self.aS/2/np.pi\
          *( (z*z+(1-z)**2)*plus+(2*z*(1-z)+(1-z)**2*(1-y))*f(z,y)*self.aux2(z,1-y))

  def dxsecCqg2AB(self,z,y):
    return self.Cqg2AB(z,y,lambda z,y: self.get_lum(z,y,'qA,gB')*self.get_jac(z,y))\

  def dxsecCqg2BA(self,z,y):
    return self.Cqg2BA(z,y,lambda z,y: self.get_lum(z,y,'gA,qB')*self.get_jac(z,y))

  def xsecCqg2AB(self):
    func=lambda z: self.quad(np.vectorize(lambda y: self.dxsecCqg2AB(z,y)),0,1)
    return self.quad(np.vectorize(func),self.tau,1)

  def xsecCqg2BA(self):
    func=lambda z: self.quad(np.vectorize(lambda y: self.dxsecCqg2BA(z,y)),0,1)
    return self.quad(np.vectorize(func),self.tau,1)

  #--combined

  def xsecCqgAB(self):
    return self.xsecCqg1AB()+self.xsecCqg2AB()

  def xsecCqgBA(self):
    return self.xsecCqg1BA()+self.xsecCqg2BA()

  def xsecCqg(self):
    return self.xsecCqgAB()+self.xsecCqgBA()

  #--combined all

  def get_xsec(self,Q2,S,Y,muF2,ilum='normal',part='full',flav=None):
    self.flav=flav
    self.ilum=ilum
    self.Q2=Q2
    self.muF2=muF2
    self.tau=Q2/S
    self.Y=Y
    self.aS=conf['alphaS'].get_alphaS(self.muF2)
    self.Nf=conf['alphaS'].get_Nf(muF2)
    aEM=self.eweak.get_alpha(Q2)
    factor=(4*np.pi*aEM**2)/(9*Q2*S)

    if   part=='qA,qbB': xsec=self.xsecCqqbAB()
    elif part=='qbA,qB': xsec=self.xsecCqqbBA()
    elif part=='q,qb':   xsec=self.xsecCqqb()
    elif part=='qA,gB':  xsec=self.xsecCqgAB()
    elif part=='gA,qB':  xsec=self.xsecCqgBA()
    elif part=='q,g':    xsec=self.xsecCqg()
    elif part=='full':   xsec=self.xsecCqqb()+self.xsecCqg()

    return factor*xsec

  def _get_xsec(self,Q2,S,Y,muF2,ixsec=2,units='pb',ilum='xspace'):
    xsec=self.get_xsec(Q2,S,Y,muF2)
    #xsec*=self.norm*self.aEM**2/Q2/S
    if   units=='mb': xsec*=0.3894
    elif units=='pb': xsec*=0.3894*1e9
    if ixsec==1:   return xsec              # dsig/dM2dY
    elif ixsec==2: return xsec*2*Q2**0.5    # dsig/dMdY

  # mellin routines 

  def gen_SIGNM(self,N,M,Q2,S,Y,muF2,part,msg='calculating...'):
    pts=len(N)
    bar=BAR(msg,pts**2)
    SIGNM=np.zeros((pts,pts),dtype=complex)
    for i in range(pts): 
      for j in range(pts): 
        self.n=N[i]
        self.m=M[j]
        real=self.get_xsec(Q2,S,Y,muF2,ilum='mell-real',part=part)
        imag=self.get_xsec(Q2,S,Y,muF2,ilum='mell-imag',part=part)
        SIGNM[i,j]=np.complex(real,imag)
        bar.next()
    bar.finish()
    return SIGNM

  def _gen_melltab(self,Q2,S,Y,muF2):
    N =self.mellin.N
    NS=np.conjugate(N)
    data={}
    for part in ['qA,qbB','qbA,qB','qA,gB','gA,qB']:
      data[part]={}
      data[part]['SIGNM']=self.gen_SIGNM(N ,N,Q2,S,Y,muF2,part,msg='calculating %s N M'%part)
      data[part]['SIGNCM']=self.gen_SIGNM(NS,N,Q2,S,Y,muF2,part,msg='calculating %s N*M'%part)
    return data

  def gen_melltab(self):
    path2dytab=conf['path2dytab']
    for k in conf['dy tabs']:
      path2dytabK='%s/%d'%(path2dytab,k)
      checkdir(path2dytabK)
      npts=len(conf['dy tabs'][k]['value'])
      for i in range(npts):
        S  = conf['dy tabs'][k]['S'][i]
        Y  = conf['dy tabs'][k]['Y'][i]
        Q2 = conf['dy tabs'][k]['Q2'][i]
        idx= conf['dy tabs'][k]['idx'][i]
        muF2=Q2
        data=self._gen_melltab(Q2,S,Y,muF2)
        fname='%s/%d.melltab'%(path2dytabK,idx)
        save(data,fname)

  def load_melltab(self):
    self.mtab={}
    path2dytab=conf['path2dytab']
    for k in conf['dy tabs']:
      path2dytabK='%s/%d'%(path2dytab,k)
      self.mtab[k]={}
      npts=len(conf['dy tabs'][k]['value'])
      bar=BAR('loading DY tables of %d'%k,npts)
      for i in range(npts):
        idx= conf['dy tabs'][k]['idx'][i]
        fname='%s/%d.melltab'%(path2dytabK,idx)
        self.mtab[k][i]=load(fname)
        bar.next()
      bar.finish() 

  def invert(self,fA,fB,sigNM,sigNCM):
    fAS=np.conjugate(fA)
    W=self.mellin.W*self.mellin.JAC
    phase2=self.mellin.phase**2
    xsec1=np.einsum('i,j,i,j,ij',W,W,fA,fB,phase2*sigNM,optimize=True)
    xsec2=np.einsum('i,j,i,j,ij',W,W,fAS,fB,sigNCM,optimize=True)
    return np.real((-2/4./np.pi**2)*(xsec1-xsec2))

  def get_mxsec(self,k,i,muF2,reaction):

    conf['pdfA'].evolve(muF2)
    conf['pdfB'].evolve(muF2)

    gA  = conf['pdfA'].storage[muF2]['g'] 
    uA  = conf['pdfA'].storage[muF2]['u'] 
    dA  = conf['pdfA'].storage[muF2]['d'] 
    sA  = conf['pdfA'].storage[muF2]['s'] 
    cA  = conf['pdfA'].storage[muF2]['c'] 
    bA  = conf['pdfA'].storage[muF2]['b'] 
    ubA = conf['pdfA'].storage[muF2]['ub']
    dbA = conf['pdfA'].storage[muF2]['db']
    sbA = conf['pdfA'].storage[muF2]['sb']
    cbA = conf['pdfA'].storage[muF2]['cb']
    bbA = conf['pdfA'].storage[muF2]['bb']

    gB  = conf['pdfB'].storage[muF2]['g'] 
    uB  = conf['pdfB'].storage[muF2]['u'] 
    dB  = conf['pdfB'].storage[muF2]['d'] 
    sB  = conf['pdfB'].storage[muF2]['s'] 
    cB  = conf['pdfB'].storage[muF2]['c'] 
    bB  = conf['pdfB'].storage[muF2]['b'] 
    ubB = conf['pdfB'].storage[muF2]['ub']
    dbB = conf['pdfB'].storage[muF2]['db']
    sbB = conf['pdfB'].storage[muF2]['sb']
    cbB = conf['pdfB'].storage[muF2]['cb'] 
    bbB = conf['pdfB'].storage[muF2]['bb'] 
    

    if reaction=='pd':
      uB=0.5*(uB+dB)
      dB=uB
      ubB=0.5*(ubB+dbB)
      dbB=ubB
    
    xsec = self.eU2*self.invert(uA  ,ubB ,self.mtab[k][i]['qA,qbB']['SIGNM'],self.mtab[k][i]['qA,qbB']['SIGNCM'])\
          +self.eD2*self.invert(dA  ,dbB ,self.mtab[k][i]['qA,qbB']['SIGNM'],self.mtab[k][i]['qA,qbB']['SIGNCM'])\
          +self.eD2*self.invert(sA  ,sbB ,self.mtab[k][i]['qA,qbB']['SIGNM'],self.mtab[k][i]['qA,qbB']['SIGNCM'])\
          +self.eU2*self.invert(ubA ,uB  ,self.mtab[k][i]['qbA,qB']['SIGNM'],self.mtab[k][i]['qbA,qB']['SIGNCM'])\
          +self.eD2*self.invert(dbA ,dB  ,self.mtab[k][i]['qbA,qB']['SIGNM'],self.mtab[k][i]['qbA,qB']['SIGNCM'])\
          +self.eD2*self.invert(sbA ,sB  ,self.mtab[k][i]['qbA,qB']['SIGNM'],self.mtab[k][i]['qbA,qB']['SIGNCM'])\
          +self.eU2*self.invert(uA  ,gB  ,self.mtab[k][i]['qA,gB']['SIGNM'] ,self.mtab[k][i]['qA,gB']['SIGNCM'] )\
          +self.eD2*self.invert(dA  ,gB  ,self.mtab[k][i]['qA,gB']['SIGNM'] ,self.mtab[k][i]['qA,gB']['SIGNCM'] )\
          +self.eD2*self.invert(sA  ,gB  ,self.mtab[k][i]['qA,gB']['SIGNM'] ,self.mtab[k][i]['qA,gB']['SIGNCM'] )\
          +self.eU2*self.invert(ubA ,gB  ,self.mtab[k][i]['qA,gB']['SIGNM'] ,self.mtab[k][i]['qA,gB']['SIGNCM'] )\
          +self.eD2*self.invert(dbA ,gB  ,self.mtab[k][i]['qA,gB']['SIGNM'] ,self.mtab[k][i]['qA,gB']['SIGNCM'] )\
          +self.eD2*self.invert(sA  ,gB  ,self.mtab[k][i]['qA,gB']['SIGNM'] ,self.mtab[k][i]['qA,gB']['SIGNCM'] )\
          +self.eU2*self.invert(gA  ,uB  ,self.mtab[k][i]['gA,qB']['SIGNM'] ,self.mtab[k][i]['gA,qB']['SIGNCM'] )\
          +self.eD2*self.invert(gA  ,dB  ,self.mtab[k][i]['gA,qB']['SIGNM'] ,self.mtab[k][i]['gA,qB']['SIGNCM'] )\
          +self.eD2*self.invert(gA  ,sB  ,self.mtab[k][i]['gA,qB']['SIGNM'] ,self.mtab[k][i]['gA,qB']['SIGNCM'] )\
          +self.eU2*self.invert(gA  ,ubB ,self.mtab[k][i]['gA,qB']['SIGNM'] ,self.mtab[k][i]['gA,qB']['SIGNCM'] )\
          +self.eD2*self.invert(gA  ,dbB ,self.mtab[k][i]['gA,qB']['SIGNM'] ,self.mtab[k][i]['gA,qB']['SIGNCM'] )\
          +self.eD2*self.invert(gA  ,sbB ,self.mtab[k][i]['gA,qB']['SIGNM'] ,self.mtab[k][i]['gA,qB']['SIGNCM'] )


    Nf=conf['alphaS'].get_Nf(muF2)
    if Nf>3: 
      xsec+= self.eU2*self.invert(cA  ,cbB  ,self.mtab[k][i]['qA,qbB']['SIGNM'],self.mtab[k][i]['qA,qbB']['SIGNCM'])\
            +self.eU2*self.invert(cbA ,cB   ,self.mtab[k][i]['qbA,qB']['SIGNM'],self.mtab[k][i]['qbA,qB']['SIGNCM'])\
            +self.eU2*self.invert(cA  ,gB   ,self.mtab[k][i]['qA,gB']['SIGNM'] ,self.mtab[k][i]['qA,gB']['SIGNCM'] )\
            +self.eU2*self.invert(cbA ,gB   ,self.mtab[k][i]['qA,gB']['SIGNM'] ,self.mtab[k][i]['qA,gB']['SIGNCM'] )\
            +self.eU2*self.invert(gA  ,cB   ,self.mtab[k][i]['gA,qB']['SIGNM'] ,self.mtab[k][i]['gA,qB']['SIGNCM'] )\
            +self.eU2*self.invert(gA  ,cbB  ,self.mtab[k][i]['gA,qB']['SIGNM'] ,self.mtab[k][i]['gA,qB']['SIGNCM'] )
                                                      
    if Nf>4:                                          
      xsec+= self.eD2*self.invert(bA  ,bbB ,self.mtab[k][i]['qA,qbB']['SIGNM'],self.mtab[k][i]['qA,qbB']['SIGNCM'])\
            +self.eD2*self.invert(bbA ,bB  ,self.mtab[k][i]['qbA,qB']['SIGNM'],self.mtab[k][i]['qbA,qB']['SIGNCM'])\
            +self.eD2*self.invert(bA  ,gB  ,self.mtab[k][i]['qA,gB']['SIGNM'] ,self.mtab[k][i]['qA,gB']['SIGNCM'] )\
            +self.eD2*self.invert(bbA ,gB  ,self.mtab[k][i]['qA,gB']['SIGNM'] ,self.mtab[k][i]['qA,gB']['SIGNCM'] )\
            +self.eD2*self.invert(gA  ,bB  ,self.mtab[k][i]['gA,qB']['SIGNM'] ,self.mtab[k][i]['gA,qB']['SIGNCM'] )\
            +self.eD2*self.invert(gA  ,bbB ,self.mtab[k][i]['gA,qB']['SIGNM'] ,self.mtab[k][i]['gA,qB']['SIGNCM'] )
    return xsec

  # hybrid mellin routines

  def gen_SIGN(self,N,Q2,S,Y,muF2,part,flav,msg='calculating...'):
    Nf=conf['alphaS'].get_Nf(Q2)
    pts=len(N)
    SIGN=np.zeros(pts,dtype=complex)
    if flav=='c'  and Nf<=3: return SIGN
    if flav=='cb' and Nf<=3: return SIGN 
    if flav=='b'  and Nf<=4: return SIGN
    if flav=='bb' and Nf<=4: return SIGN
    for i in range(pts): 
      self.n=N[i]
      self.flav=flav
      real=self.get_xsec(Q2,S,Y,muF2,ilum='hybrid-real',part=part)
      imag=self.get_xsec(Q2,S,Y,muF2,ilum='hybrid-imag',part=part)
      SIGN[i]=np.complex(real,imag)
    return SIGN

  def _gen_melltab_hybrid(self,Q2,S,Y,muF2):
    N =self.mellin.N
    data={}

    data['qA,qbB']={}
    for flav in ['ub','db','sb','cb','bb']:
      data['qA,qbB'][flav]=self.gen_SIGN(N,Q2,S,Y,muF2,'qA,qbB',flav,msg='calculating %s %s'%('qA,qbB',flav))

    data['qbA,qB']={}
    for flav in ['u','d','s','c','b']:
      data['qbA,qB'][flav]=self.gen_SIGN(N,Q2,S,Y,muF2,'qbA,qB',flav,msg='calculating %s %s'%('qbA,qB',flav))

    data['qA,gB']={}
    for flav in ['g']:
      data['qA,gB'][flav]=self.gen_SIGN(N,Q2,S,Y,muF2,'qA,gB','g',msg='calculating %s %s'%('qA,gB',flav))

    data['gA,qB']={}
    for flav in ['u','d','s','c','b','ub','db','sb','cb','bb']:
      data['gA,qB'][flav]=self.gen_SIGN(N,Q2,S,Y,muF2,'gA,qB',flav,msg='calculating %s %s'%('gA,qB',flav))

    return data

  def gen_melltab_hybrid(self):
    path2dytab=conf['path2dytab-hybrid']
    for k in conf['dy-pion tabs']:
      path2dytabK='%s/%d'%(path2dytab,k)
      checkdir(path2dytabK)
      for i in range(len(conf['dy-pion tabs'][k]['idx'])):
        print '%s: %i/%i'%(k,i,len(conf['dy-pion tabs'][k]['idx']))
        idx=conf['dy-pion tabs'][k]['idx'][i]
        S  = conf['dy-pion tabs'][k]['S'][i]
        Y  = conf['dy-pion tabs'][k]['Y'][i]
        Q2 = conf['dy-pion tabs'][k]['Q2'][i]
        muF2=Q2
        data=self._gen_melltab_hybrid(Q2,S,Y,muF2)
        fname='%s/%d.melltab'%(path2dytabK,idx)
        save(data,fname)

  def load_melltab_hybrid(self):
    self.melltab={}
    path2dytab=conf['path2dytab-hybrid']
    for k in conf['dy-pion tabs']:
      path2dytabK='%s/%d'%(path2dytab,k)
      self.melltab[k]={}
      for idx in conf['dy-pion tabs'][k]['idx']:
        fname='%s/%d.melltab'%(path2dytabK,idx)
        self.melltab[k][idx]=load(fname)

  def load_nn_hybrid(self):
    self.nn={}
    path2nntab=conf['path2nntabs']
    initx=True
    self.yscaler={}
    for part in ['qA,qbB','qbA,qB','qA,gB','gA,qB']:
      self.nn[part]={}
      self.yscaler[part]={}
      if part=='qA,qbB': flavs=['ub','db','sb','cb','bb']
      if part=='qbA,qB': flavs=['u','d','s','c','b']
      if part=='qA,gB': flavs=['g']
      if part=='gA,qB': flavs=['u','d','s','c','b','ub','db','sb','cb','bb']
      for f in flavs:
          fname='%s/30001/%s/%s.json'%(path2nntab,part,f) #--the 30001 is for a specific dataset!
          nn=load(fname)
          self.nn[part][f]={}
          self.nn[part][f]['predict']=nn['model'].predict
          if initx:
              xtrain=nn['x_train_orig'] #--because the xtrain is the same for all parts and flavors
              self.xscaler=MinMaxScaler()
              self.xscaler.fit_transform(xtrain)
              initx=False
          ytrain=nn['y_train_orig']
          self.yscaler[part][f]=MinMaxScaler()
          self.yscaler[part][f].fit_transform(ytrain)

  def get_xsec_mell_hybrid(self,k,i,Q2,part,flav=None):
    Nf=conf['alphaS'].get_Nf(Q2)
    idx=conf['dy-pion tabs'][k]['idx'][i]
    data=self.melltab[k][idx]
    conf['pdf-pion'].evolve(Q2)
    PDFA=conf['pdf-pion'].storage[Q2]

    """
    Put in the moments for pdfA (the pion)
    through calling the NPDF class and get_moments(flav,Q2) function
    
    """

    if part=='qA,qbB':
        if flav=='ub': mult=self.eU2*PDFA['u']
        if flav=='db': mult=self.eD2*PDFA['d']
        if flav=='sb': mult=self.eD2*PDFA['s']
        if flav=='cb': mult=self.eU2*PDFA['c']
        if flav=='bb': mult=self.eD2*PDFA['b']
    if part=='qbA,qB':
        if flav=='u': mult=self.eU2*PDFA['ub']
        if flav=='d': mult=self.eD2*PDFA['db']
        if flav=='s': mult=self.eD2*PDFA['sb']
        if flav=='c': mult=self.eU2*PDFA['cb']
        if flav=='b': mult=self.eD2*PDFA['bb']
    if part=='qA,gB':
        mult=self.eU2*(PDFA['u']+PDFA['ub'])\
            +self.eD2*(PDFA['d']+PDFA['db'])\
            +self.eD2*(PDFA['s']+PDFA['sb'])
        if Nf>3:
            mult+=self.eU2*(PDFA['c']+PDFA['cb'])
        if Nf>4:
            mult+=self.eD2*(PDFA['b']+PDFA['bb'])
    if part=='gA,qB':
        if flav=='u':  mult=self.eU2*PDFA['g']
        if flav=='ub': mult=self.eU2*PDFA['g']
        if flav=='d':  mult=self.eD2*PDFA['g']
        if flav=='db': mult=self.eD2*PDFA['g']
        if flav=='s':  mult=self.eD2*PDFA['g']
        if flav=='sb': mult=self.eD2*PDFA['g']
        if flav=='c':  mult=self.eU2*PDFA['g']
        if flav=='cb': mult=self.eU2*PDFA['g']
        if flav=='b':  mult=self.eD2*PDFA['g']
        if flav=='bb': mult=self.eD2*PDFA['g']

    if part!='full': 
        xsec=mult*data[part][flav]
        checkdir('%s/data/%s/%s/%s'%(conf['path2ml4jam'],k,part,flav))
        save(data[part][flav],'%s/data/%s/%s/%s/%smell.dat'%(conf['path2ml4jam'],k,part,flav,i))

    elif part=='full':
        xsec = self.eU2*PDFA['u']  * data['qA,qbB']['ub']\
              +self.eD2*PDFA['d']  * data['qA,qbB']['db']\
              +self.eD2*PDFA['s']  * data['qA,qbB']['sb']\
              +self.eU2*PDFA['ub'] * data['qbA,qB']['u']\
              +self.eD2*PDFA['db'] * data['qbA,qB']['d']\
              +self.eD2*PDFA['sb'] * data['qbA,qB']['s']\
              +self.eU2*PDFA['u']  * data['qA,gB']['g']\
              +self.eD2*PDFA['d']  * data['qA,gB']['g']\
              +self.eD2*PDFA['s']  * data['qA,gB']['g']\
              +self.eU2*PDFA['ub'] * data['qA,gB']['g']\
              +self.eD2*PDFA['db'] * data['qA,gB']['g']\
              +self.eD2*PDFA['sb'] * data['qA,gB']['g']\
              +self.eU2*PDFA['g']  * data['gA,qB']['u']\
              +self.eD2*PDFA['g']  * data['gA,qB']['d']\
              +self.eD2*PDFA['g']  * data['gA,qB']['s']\
              +self.eU2*PDFA['g']  * data['gA,qB']['ub']\
              +self.eD2*PDFA['g']  * data['gA,qB']['db']\
              +self.eD2*PDFA['g']  * data['gA,qB']['sb']

        if Nf>3: 
          xsec+= self.eU2*PDFA['c']  * data['qA,qbB']['cb']\
                +self.eU2*PDFA['cb'] * data['qbA,qB']['c']\
                +self.eU2*PDFA['c']  * data['qA,gB']['g']\
                +self.eU2*PDFA['cb'] * data['qA,gB']['g']\
                +self.eU2*PDFA['g']  * data['gA,qB']['c']\
                +self.eU2*PDFA['g']  * data['gA,qB']['cb']

        if Nf>4: 
          xsec+= self.eD2*PDFA['b']  * data['qA,qbB']['bb']\
                +self.eD2*PDFA['bb'] * data['qbA,qB']['b']\
                +self.eD2*PDFA['b']  * data['qA,gB']['g']\
                +self.eD2*PDFA['bb'] * data['qA,gB']['g']\
                +self.eD2*PDFA['g']  * data['gA,qB']['b']\
                +self.eD2*PDFA['g']  * data['gA,qB']['bb']
        checkdir('%s/data/%s/%s'%(conf['path2ml4jam'],k,part))
        save(xsec,'%s/data/%s/%s/%sxsecmell.dat'%(conf['path2ml4jam'],k,part,i))

    return self.mellin_ext.invert(1,xsec) # x=1 because x**-N is inside data[..][..]

  def get_xsec_mell_hybrid_nn(self,k,i,Q2,part,flav=None):
    Nf=conf['alphaS'].get_Nf(Q2)
    conf['pdf-pion'].evolve(Q2)
    PDFA=conf['pdf-pion'].storage[Q2]

    #--include machine learning
    N=conf['mellin-pion'].N
    xF=np.ones(len(N))*conf['dy-pion tabs'][k]['xF'][i]
    ReN=N.real
    ImN=N.imag
    x_vals=np.array([xF,ReN,ImN]).T
    x_vals=self.xscaler.transform(x_vals)

    data={}
    if part!='full':
        data[part]={}
        result=self.nn[part][flav]['predict'](x_vals)
        result=self.yscaler[part][flav].inverse_transform(result).T
        data[part][flav]=result[0]+1j*result[1]
        checkdir('%s/data/%s/%s/%s'%(conf['path2ml4jam'],k,part,flav))
        save(data[part][flav],'%s/data/%s/%s/%s/%snn.dat'%(conf['path2ml4jam'],k,part,flav,i))

    """
    Put in the moments for pdfA (the pion)
    through calling the NPDF class and get_moments(flav,Q2) function
    
    """

    if part=='qA,qbB':
        if flav=='ub': mult=self.eU2*PDFA['u']
        if flav=='db': mult=self.eD2*PDFA['d']
        if flav=='sb': mult=self.eD2*PDFA['s']
        if flav=='cb': mult=self.eU2*PDFA['c']
        if flav=='bb': mult=self.eD2*PDFA['b']
    if part=='qbA,qB':
        if flav=='u': mult=self.eU2*PDFA['ub']
        if flav=='d': mult=self.eD2*PDFA['db']
        if flav=='s': mult=self.eD2*PDFA['sb']
        if flav=='c': mult=self.eU2*PDFA['cb']
        if flav=='b': mult=self.eD2*PDFA['bb']
    if part=='qA,gB':
        mult=self.eU2*(PDFA['u']+PDFA['ub'])\
            +self.eD2*(PDFA['d']+PDFA['db'])\
            +self.eD2*(PDFA['s']+PDFA['sb'])
        if Nf>3:
            mult+=self.eU2*(PDFA['c']+PDFA['cb'])
        if Nf>4:
            mult+=self.eD2*(PDFA['b']+PDFA['bb'])
    if part=='gA,qB':
        if flav=='u':  mult=self.eU2*PDFA['g']
        if flav=='ub': mult=self.eU2*PDFA['g']
        if flav=='d':  mult=self.eD2*PDFA['g']
        if flav=='db': mult=self.eD2*PDFA['g']
        if flav=='s':  mult=self.eD2*PDFA['g']
        if flav=='sb': mult=self.eD2*PDFA['g']
        if flav=='c':  mult=self.eU2*PDFA['g']
        if flav=='cb': mult=self.eU2*PDFA['g']
        if flav=='b':  mult=self.eD2*PDFA['g']
        if flav=='bb': mult=self.eD2*PDFA['g']

    if part!='full': xsec=mult*data[part][flav]

    elif part=='full':
        for p in ['qA,qbB','qbA,qB','qA,gB','gA,qB']:
          data[p]={}
          if p=='qA,qbB': flavs=['ub','db','sb','cb','bb']
          if p=='qbA,qB': flavs=['u','d','s','c','b']
          if p=='qA,gB': flavs=['g']
          if p=='gA,qB': flavs=['u','d','s','c','b','ub','db','sb','cb','bb']
          for f in flavs:
              result=self.nn[p][f]['predict'](x_vals)
              result=self.yscaler[p][f].inverse_transform(result).T
              data[p][f]=result[0]+1j*result[1]
    
        xsec = self.eU2*PDFA['u']  * data['qA,qbB']['ub']\
              +self.eD2*PDFA['d']  * data['qA,qbB']['db']\
              +self.eD2*PDFA['s']  * data['qA,qbB']['sb']\
              +self.eU2*PDFA['ub'] * data['qbA,qB']['u']\
              +self.eD2*PDFA['db'] * data['qbA,qB']['d']\
              +self.eD2*PDFA['sb'] * data['qbA,qB']['s']\
              +self.eU2*PDFA['u']  * data['qA,gB']['g']\
              +self.eD2*PDFA['d']  * data['qA,gB']['g']\
              +self.eD2*PDFA['s']  * data['qA,gB']['g']\
              +self.eU2*PDFA['ub'] * data['qA,gB']['g']\
              +self.eD2*PDFA['db'] * data['qA,gB']['g']\
              +self.eD2*PDFA['sb'] * data['qA,gB']['g']\
              +self.eU2*PDFA['g']  * data['gA,qB']['u']\
              +self.eD2*PDFA['g']  * data['gA,qB']['d']\
              +self.eD2*PDFA['g']  * data['gA,qB']['s']\
              +self.eU2*PDFA['g']  * data['gA,qB']['ub']\
              +self.eD2*PDFA['g']  * data['gA,qB']['db']\
              +self.eD2*PDFA['g']  * data['gA,qB']['sb']

        if Nf>3: 
          xsec+= self.eU2*PDFA['c']  * data['qA,qbB']['cb']\
                +self.eU2*PDFA['cb'] * data['qbA,qB']['c']\
                +self.eU2*PDFA['c']  * data['qA,gB']['g']\
                +self.eU2*PDFA['cb'] * data['qA,gB']['g']\
                +self.eU2*PDFA['g']  * data['gA,qB']['c']\
                +self.eU2*PDFA['g']  * data['gA,qB']['cb']

        if Nf>4: 
          xsec+= self.eD2*PDFA['b']  * data['qA,qbB']['bb']\
                +self.eD2*PDFA['bb'] * data['qbA,qB']['b']\
                +self.eD2*PDFA['b']  * data['qA,gB']['g']\
                +self.eD2*PDFA['bb'] * data['qA,gB']['g']\
                +self.eD2*PDFA['g']  * data['gA,qB']['b']\
                +self.eD2*PDFA['g']  * data['gA,qB']['bb']
        checkdir('%s/data/%s/%s'%(conf['path2ml4jam'],k,part))
        save(xsec,'%s/data/%s/%s/%sxsecnn.dat'%(conf['path2ml4jam'],k,part,i))

    return self.mellin_ext.invert(1,xsec) # x=1 because x**-N is inside data[..][..]

if __name__=='__main__':

  import os
  from qcdlib import aux,mellin,alphaS,eweak
  from fakepdf import FAKEPDF

  conf['Q20']   = 1.0
  conf['alphaSmode']='backward'
  conf['mode']='truncated'
  conf['order']='NLO'
  conf['scheme']='ZMVFS'
  conf['path2dytab']='%s/grids/dytab'%os.environ['FITPACK']
  conf['aux']=aux.AUX()
  conf['mellin']=mellin.MELLIN(npts=4)
  conf['alphaS']=alphaS.ALPHAS()
  conf['eweak']=eweak.EWEAK()
  conf['pdfA']=FAKEPDF()
  conf['pdfB']=FAKEPDF()

  ############################################
  # tests

  #Q2=10.0
  #S=32.0**2
  #Y=0.0
  #muF2=Q2
  #dy=DY()
  #print dy.get_xsec(Q2,S,Y,muF2,ilum='normal',part='full')

  #############################################
  # precalcs

  from qcdlib import aux
  from reader import READER  
  import time


  conf['aux']=aux.AUX()
  conf['datasets']={}
  conf['datasets']['dy']={}
  conf['datasets']['dy']['xlsx']={}
  conf['datasets']['dy']['xlsx'][10001]='dy/expdata/10001.xlsx'
  conf['datasets']['dy']['xlsx'][10002]='dy/expdata/10002.xlsx'
  conf['datasets']['dy']['norm']={}
  conf['datasets']['dy']['filters']=[]
  conf['datasets']['dy']['filters'].append("Q2>1") 


  conf['dy tabs']=READER().load_data_sets('dy')
  dy=DY()
  #dy.gen_melltab()
  dy.load_melltab()

  t1=time.time()
  for k in conf['dy tabs']:
    npts=len(conf['dy tabs'][k]['value'])
    for i in range(npts):
      Q2 = conf['dy tabs'][k]['Q2'][i]
      reaction = conf['dy tabs'][k]['reaction'][i]
      approx = dy.get_mxsec(k,i,Q2,reaction)
      #print i
      S  = conf['dy tabs'][k]['S'][i]
      Y  = conf['dy tabs'][k]['Y'][i]
      muF2=Q2
      exact =  dy.get_xsec(Q2,S,Y,muF2,ilum='normal',part='full')
      rel_err=abs((approx-exact)/exact)*100
      print rel_err


  t2=time.time()
  print t2-t1











