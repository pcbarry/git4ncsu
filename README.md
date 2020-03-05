# git4ncsu
Repository for tutorials of physical examples of machine learning

Use the jupyter notebooks (Fitter.ipynb and ML.ipynb) to execute all the code.

# Descriptions of Directories
## database:
- contains the experimental data with physical cross sections and kinematics

## dy:
- fakepdf.py: object of toy PDF to use for educational purposes
- reader.py:  allows us to read in the DY experimental data and apply kinematic cuts
- theory.py:  contains all the theory for calculating a cross section

## melltabs:
- contains the mellin tables on which to train.  Ping me if you have questions about what's inside

## nptabs:
- contains the mellin tables in the numpy array format.  Needed for machine learning

## qcdlib:
- alphaS.py: calculate the strong coupling recursively
- aux.py:    define constants and masses
- eweak.py:  calculate the electroweak coupling as a funcion of scale
- mellin.py: define the mellin contour and calcuate inversions

## tools:
- config.py: used mostly for the initialization of the configuration
- reader.py: main class for the reader, different datasets are initialized as this class
- tools.py:  various tools for saving, checking the directory, and loading
