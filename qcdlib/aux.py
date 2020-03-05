from mpmath import fp
from scipy.special import gamma
import numpy as np
from tools.config import conf

class AUX:

  def __init__(self):

    self.set_constants()
    self.set_masses()
    self.set_couplings()

  def set_constants(self):
    self.CA=3.0
    self.CF=4.0/3.0
    self.TR=0.5
    self.TF=0.5
    self.euler=fp.euler 

  def set_masses(self):

    self.me   = 0.000511
    self.mmu  = 0.105658
    self.mtau = 1.77684
    self.mu   = 0.055
    self.md   = 0.055
    self.ms   = 0.2

    # PDG
    self.mc   = 1.28
    self.mb   = 4.18
    
    if 'cj masses' in conf:
      if conf['cj masses']:
        self.mc=1.3
        self.mb=4.5

    self.mZ   = 91.1876
    self.mW   = 80.398
    self.M    = 0.93891897
    self.Mpi  = 0.13803
    self.Mk   = 0.493677
    self.Mdelta = 1.232

    self.me2   = self.me**2 
    self.mmu2  = self.mmu**2 
    self.mtau2 = self.mtau**2
    self.mu2   = self.mu**2  
    self.md2   = self.md**2  
    self.ms2   = self.ms**2  
    self.mc2   = self.mc**2  
    self.mb2   = self.mb**2  
    self.mZ2   = self.mZ**2  
    self.mW2   = self.mW**2  
    self.M2    = self.M**2  
    self.Mpi2  = self.Mpi**2  
    self.Mdelta2=self.Mdelta**2

  def set_couplings(self):

    self.c2w = self.mW2/self.mZ2
    self.s2w = 1.0-self.c2w
    self.s2wMZ = 0.23116
    self.alfa  = 1/137.036
    if 'alphaS(MZ)' in conf:
      self.alphaSMZ=conf['alphaS(MZ)']
    else:
      self.alphaSMZ = 0.118
    self.GF = 1.1663787e-5   # 1/GeV^2
  
  def _get_psi(self,i,N):
    return fp.psi(i,complex(N.real,N.imag))

  def get_psi(self,i,N):
    return np.array([self._get_psi(i,n) for n in N],dtype=complex)

  def beta(self,a,b):
    return gamma(a)*gamma(b)/gamma(a+b)


