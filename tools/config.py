import sys
conf = {}
options = {}


def load_config(fname):

    L=open(fname).readlines()
    D = {}
    for l in L:
        try:
            exec l in D#l.replace('<<','').replace('>>','') in D
        except:
            print('ERR at the input.py. Look for %s'%l)
            sys.exit()   
 
    conf.update(D['conf'])



