from scipy.special import gamma
from tools.config import conf

class FAKEPDF :

  def __init__ (self):

    self.mellin=conf['mellin']
    self.params={}
    self.params['g'] =[1,-0.4,6,0,0]
    self.params['u'] =[1,-0.5,3,0,0]
    self.params['d'] =[1,-0.5,3,0,0]
    self.params['s'] =[1,-0.5,6,0,0]
    self.params['c'] =[1,-0.5,6,0,0]
    self.params['b'] =[1,-0.5,6,0,0]
    self.params['ub']=[1,-0.5,6,0,0]
    self.params['db']=[1,-0.5,6,0,0]
    self.params['sb'] =[1,-0.5,6,0,0]
    self.params['cb'] =[1,-0.5,6,0,0]
    self.params['bb'] =[1,-0.5,6,0,0]

    self.storage={}

  def _get_pdf(self,x,flav): 
    N,a,b,c,d=self.params[flav]
    return N*x**a*(1-x)**b*(1+c*x**0.5+d*x)

  def get_pdfs(self,x,Q2): 
    pdfs=[]
    for flav in ['g','u','ub','d','db','s','sb','c','cb','b','bb']:
      pdfs.append(self._get_pdf(x,flav))
    return pdfs 

  def beta(self,a,b):
    return gamma(a)*gamma(b)/gamma(a+b)

  def get_moments(self,flav,N=None):
    """
     if N==None: then parameterization is to be use to compute moments along mellin contour
    else the Nth moment is returned
    """
    if N==None: N=self.mellin.N
    norm,a,b,c,d=self.params[flav]
    return norm*(self.beta(N+a,b+1)+c*self.beta(N+a+0.5,b+1)+d*self.beta(N+a+1,b+1))

  def evolve(self,Q2):

    if Q2 not in self.storage:
      self.storage[Q2]={}
      self.storage[Q2]['g'] =self.get_moments('g')
      self.storage[Q2]['u'] =self.get_moments('u')
      self.storage[Q2]['d'] =self.get_moments('d')
      self.storage[Q2]['s'] =self.get_moments('s')
      self.storage[Q2]['c'] =self.get_moments('c')
      self.storage[Q2]['b'] =self.get_moments('b')
      self.storage[Q2]['ub']=self.get_moments('ub')
      self.storage[Q2]['db']=self.get_moments('db')
      self.storage[Q2]['sb']=self.get_moments('sb')
      self.storage[Q2]['cb']=self.get_moments('cb')
      self.storage[Q2]['bb']=self.get_moments('bb')


