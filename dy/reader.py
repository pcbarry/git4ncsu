#!/usr/bin/env python
import numpy as np
import pandas as pd
from tools.reader import _READER
from tools.config import conf

class READER(_READER):

  def __init__(self):
    self.aux=conf['aux']

  def get_tau(self,tab):
    rtau=0.5*(tab['rtaumin'].values+tab['rtaumax'].values)
    tab['tau']=pd.Series(rtau**2,index=tab.index)
    return tab

  def get_xF(self,tab):
    xF=0.5*(tab['xFmin'].values+tab['xFmax'].values)
    tab['xF']=pd.Series(xF,index=tab.index)
    return tab

  def get_Y(self,tab):
    Y=np.arcsinh(0.5*tab['xF']/tab['tau']**0.5)
    tab['Y']=pd.Series(Y,index=tab.index)
    return tab

  def get_S(self,tab):
    S=2*tab.E*self.aux.M
    tab['S']=pd.Series(S,index=tab.index)
    return tab

  def get_Q2(self,tab):
    Q2=tab['tau'].values*tab['S'].values
    tab['Q2']=pd.Series(Q2,index=tab.index)
    return tab

  def get_xpi(self,tab):
    XF=tab['xF'].values
    tau=tab['tau'].values
    xpi=0.5*(XF+(XF**2+4*tau)**0.5)
    tab['xpi']=pd.Series(xpi,index=tab.index)
    return tab

  def get_Jac(self,tab):
    if tab.obs.values[0]=='dsig/drtau/dxF':
      Jac=tab['S']/np.cosh(tab['Y'])
    tab['Jac']=pd.Series(Jac,index=tab.index)
    return tab

  def get_units(self,tab):
    units=np.ones(len(tab.index))
    if tab.units.values[0]=='nb':
      units*=0.389379e6
    tab['Units']=pd.Series(units,index=tab.index)
    return tab

  def get_idx(self,tab):
    tab['idx']=pd.Series(tab.index,index=tab.index)
    return tab

  def modify_table(self,tab):
    tab=self.get_tau(tab)
    tab=self.get_xF(tab)
    tab=self.get_Y(tab)
    tab=self.get_S(tab)
    tab=self.get_Q2(tab)
    tab=self.get_xpi(tab)
    tab=self.get_Jac(tab)
    tab=self.get_units(tab)
    tab=self.apply_cuts(tab)
    tab=self.get_idx(tab)
    return tab


if __name__ == "__main__":

    from qcdlib import aux

    conf['aux']=aux.AUX()
    conf['datasets']={}
    conf['datasets']['dy']={}
    conf['datasets']['dy']['xlsx']={}
    conf['datasets']['dy']['xlsx'][10001]='dy/expdata/10001.xlsx'
    conf['datasets']['dy']['xlsx'][10002]='dy/expdata/10002.xlsx'
    conf['datasets']['dy']['norm']={}
    conf['datasets']['dy']['filters']=[]
    conf['datasets']['dy']['filters'].append("Q2>1") 

    TAB=READER().load_data_sets('dy')





