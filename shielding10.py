import numpy as np
import pandas as pd
from numpy import genfromtxt
from gekko import GEKKO
import joblib
from gekko import ML
import csv

class GekkoDense():
    def __init__(self,layer,m=None):
        n_in,n_out,W,b,activation = layer
        self.weights = W
        self.bias = b
        if m != None:
            self.hookGekko(m)
            
        self.af = activation
        
    #hooks to gekko
    def hookGekko(self,m):
        self.m = m
        
    def activation(self,x,skipAct=False):
        af = self.af
        if skipAct:
            return x
        if af == 'relu':                                      
            return self.m.max3(0,x)                           
                                                              
        elif af == 'sigmoid':                                 
            return 1/(1 + self.m.exp(-x))                     
        elif af == 'tanh':                                    
            return self.m.tanh(x)                             
        elif af == 'softsign':                                
            return x / (self.m.abs2(x) + 1)                   
        elif af == 'exponential':                             
            return self.m.exp(x)                              
        elif af == 'softplus':                                
            return self.m.log(self.m.exp(x) + 1)              
        elif af == 'elu':                                     
            alpha = 1.0                                       
            return self.m.if3(x,alpha * (self.m.exp(x) - 1),x)
        elif af == 'selu':                                    
            alpha = 1.67326324                                
            scale = 1.05070098                                
            return self.m.if3(x,scale*alpha*(self.m.exp(x)-1),scale*x)
        else:
            return x
        
    def forward(self,x,skipAct=False):
        n = self.weights.shape[1]
        return [self.m.Intermediate(self.activation(self.m.sum(self.weights[:,i] * x) + self.bias[i],skipAct)) for i in range(n)]

        return lNext
    def __call__(self,x,skipAct=False):
        return self.forward(x,skipAct)

#decompose the model
class gekko_sk():
    def __init__(self,model,m):
        self.m = m

        self.W = model.coefs_
        self.b = model.intercepts_
        self.hidden_layer_sizes = model.hidden_layer_sizes
        self.n_in = model.n_features_in_
        self.n_out = model.n_outputs_
        self.activation = model.activation

        self.layers = []
        if len(model.hidden_layer_sizes) == 0:
            layer = [self.n_in,self.n_out,self.W[0],self.b[0],self.activation]
            self.layers.append(GekkoDense(layer,m))
        else:
            layer = [self.n_in,self.hidden_layer_sizes[0],self.W[0],self.b[0],self.activation]
            self.layers.append(GekkoDense(layer,m))
            for i in range(1,len(self.hidden_layer_sizes)):
                layer = [self.hidden_layer_sizes[i],self.hidden_layer_sizes[i+1],self.W[i],self.b[i],self.activation]
                self.layers.append(GekkoDense(layer,m))
            layer = [self.hidden_layer_sizes[-1],self.n_out,self.W[-1],self.b[-1],self.activation]
            self.layers.append(GekkoDense(layer,m))
    
    def forward(self,x):
        l = x
        skipAct = False
        for i in range(len(self.layers)):
            if i==len(self.layers) - 1:
                skipAct = True
            l = self.layers[i](l,skipAct)
        return l
            
    def __call__(self,x):
        return self.forward(x)
    
    def predict(self,x,return_std=False):
        return self.forward(x)[0]

# Definitions
allCols = [  'MaterialIDs','Density','True Radius (m)','Radius (m)', 'CellCount', 'Cross-Sectional Area (m^2)','Volume (m^3)','Mass (kg)',\
             'LogXSgamABe6','LogXSgamABe7','LogXSgamPROe-2','LogXSgamPROe-1','LogXSgamPROe4','LogXSgamPROe5',\
             'LogXSgamPROe6','LogXSgamPROe7','LogXSalphaPROe-2','LogXSalphaPROe-1','LogXSalphaPROe4','LogXSalphaPROe5',\
             'LogXSalphaPROe6','LogXSalphaPROe7','LogXSelasticE-2','LogXSelasticE-1','LogXSelasticE4','LogXSelasticE5',\
             'LogXSelasticE6','LogXSelasticE7','LogPeakElastic','LogPeakAlpha','LogPeakGA','LogPeakGE',\
             'LogXSgamABe6Nave','LogXSgamABe7Nave','LogXSgamPROe-2Nave','LogXSgamPROe-1Nave','LogXSgamPROe4Nave','LogXSgamPROe5Nave',\
             'LogXSgamPROe6Nave','LogXSgamPROe7Nave','LogXSalphaPROe-2Nave','LogXSalphaPROe-1Nave','LogXSalphaPROe4Nave','LogXSalphaPROe5Nave',\
             'LogXSalphaPROe6Nave','LogXSalphaPROe7Nave','LogXSelasticE-2Nave','LogXSelasticE-1Nave','LogXSelasticE4Nave','LogXSelasticE5Nave',\
             'LogXSelasticE6Nave','LogXSelasticE7Nave','LogPeakElasticNave','LogPeakAlphaNave','LogPeakGANave','LogPeakGENave',\
             'LineRate','LineCost','TotalCost','TotalMass']
# ELements to drop to get just features: [0,5,6,7,9:32,39,57:60]
colsToDrop = ['MaterialIDs', 'CellCount', 'Cross-Sectional Area (m^2)',\
              'LogXSgamABe6','LogXSgamABe7','LogXSgamPROe-2','LogXSgamPROe-1','LogXSgamPROe4','LogXSgamPROe5',\
              'LogXSgamPROe6','LogXSgamPROe7','LogXSalphaPROe-2','LogXSalphaPROe-1','LogXSalphaPROe4','LogXSalphaPROe5',\
              'LogXSalphaPROe6','LogXSalphaPROe7','LogXSelasticE-2','LogXSelasticE-1','LogXSelasticE4','LogXSelasticE5',\
              'LogXSelasticE6','LogXSelasticE7','LogPeakElastic','LogPeakAlpha','LogPeakGA','LogPeakGE',\
              'LineRate','LineCost','TotalCost','TotalMass']

features = ['Density','True Radius (m)','Radius (m)','Mass (kg)','LogXSgamABe6Nave',\
            'LogXSgamABe7Nave','LogXSgamPROe-2Nave',\
            'LogXSgamPROe-1Nave','LogXSgamPROe4Nave','LogXSgamPROe5Nave','LogXSgamPROe7Nave',\
            'LogXSalphaPROe-2Nave','LogXSalphaPROe-1Nave',\
            'LogXSalphaPROe4Nave','LogXSalphaPROe5Nave','LogXSalphaPROe6Nave','LogXSalphaPROe7Nave','LogXSelasticE-2Nave',\
            'LogXSelasticE-1Nave','LogXSelasticE4Nave',\
            'LogXSelasticE5Nave','LogXSelasticE6Nave','LogXSelasticE7Nave','LogPeakElasticNave','LogPeakAlphaNave',\
            'LogPeakGANave','LogPeakGENave']

fullOBJs =\
[
# lead
[0,11300,2.2,1.00E-07,3.00E-05,5.00E-04,1.00E-04,2.00E-06,1.00E+00,\
                2.50E-04,2.00E-04,1.00E+08,1.00E+08,1.00E+08,1.00E+08,\
                1.00E+08,1.00E-05,1.10E+01,1.10E+01,1.10E+01,8.00E+01,\
                1.10E+01,2.60E+00,45,8.00E-08,3.00E-03,1],
# B4C
[1,2500,17.967,4.00E-09,8.02E-07,4.01E-01,1.60E-01,4.02E-04,1.63E-04,\
             5.80E-05,3.40E-05,4.80E+03,1.60E+03,3.20E+00,8.00E-01,\
             1.60E-01,4.80E-02,3.00E+00,2.66E+00,2.74E+00,3.30E+00,\
             2.14E+00,7.70E-01,4.72E+01,1.60E+05,4.00E-06,1.35E+01],
# HDPE
[2,998.88,2.28,2.03E-09,1.30376E-06,0.012736752,0.005094701,1.43664E-05,9.98363E-06,\
              1.76293E-05,2.97779E-05,1.33E+02,4.44E+01,8.89E-02,2.22E-02,\
              4.44E-03,3.98E-02,1.69E+00,1.64E+00,1.65E+00,1.53E+00,\
              9.24E-01,2.30E-01,1.25E+03,4.44E+03,1.96E-03,1.32E+01],
# BC
[3,2019.5,2.165,4.75E-09,2.00E-06,2.53E-01,1.01E-01,2.55E-04,1.09E-04,\
           5.50E-05,5.50E-05,3.00E+03,1.00E+03,2.00000001E+00,5.00E-01,\
           1.00E-01,7.50E-02,3.75E+00,3.50E+00,3.55E+00,3.75E+00,\
           2.35E+00,7.25E-01,5.95E+01,1.00E+05,4.00E-06,1.01E+01],
# tungsten
[4,19300,30.3,2.25E-05,2.83E-03,2.603073,0.8227445,1.93E-01,1.70E-01,\
          8.07E-02,7.19E-04,0.00000001E+00,0.00000001E+00,0.00000001E+00,0.00000001E+00,\
          0.00000001E+00,1.98E-05,7.38606,7.33006,11.3226,8.55177,\
          4.17167,2.48124,1.00E+04,7.50E-03,2.50E-02,1.00E+03],
# depleated Uranium
[5,19000,117.04,2.96E-05,0.000182087,4.261824,1.3647543,1.58E+00,1.79E-01,\
           1.28E-01,9.17E-04,0.00000001E+00,0.00000001E+00,0.00000001E+00,0.00000001E+00,\
           0.00000001E+00,1.00E-06,9.33244,9.283234,12.2224,11.107,\
           4.25407,2.72168,1.00E+04,5.00E-02,7.00E-03,1.00E+04]

# Boreated water
# [6,1095.65,1.5006544,5.55E-09,1.37E-06,9.67E-01,2.44E-01,1.12E-03,1.79E-04,\
#              1.11E-04,9.30E-05,1.75E+02,5.82E+01,1.16E-01,2.91E-02,\
#              5.82E-03,8.65E-02,6.46E+01,3.77E+01,3.14E+01,2.19E+01,\
#              9.23E+00,1.86E+00,1.25E+03,5.82E+03,5.84E-06,1.32E+01]
]


maxesForVarsOnly = [19300,-8.691793984,-6.096042291,-3.301029996,-4,-5.698970004,-5.000711535,-4.526106548,-1000,\
                    -1000,-1000,-1000,-1000,-5.99910194,1.810113597,1.576092407,1.496538635,1.903089987,1.041392685,\
                    -0.638980905,4,-7.096910013,-5.397940009,4]


IDs              = []
densities        = []
costs            = []
LogXSgamABe6     = []
LogXSgamABe7     = []
LogXSgamPROe_2   = []
LogXSgamPROe_1   = []
LogXSgamPROe4    = []
LogXSgamPROe5    = []
LogXSgamPROe6    = []
LogXSgamPROe7    = []
LogXSalphaPROe_2 = []
LogXSalphaPROe_1 = []
LogXSalphaPROe4  = []
LogXSalphaPROe5  = []
LogXSalphaPROe6  = []
LogXSalphaPROe7  = []
LogXSelasticE_2  = []
LogXSelasticE_1  = []
LogXSelasticE4   = []
LogXSelasticE5   = []
LogXSelasticE6   = []
LogXSelasticE7   = []
LogPeakElastic   = []
LogPeakAlpha     = []
LogPeakGA        = []
LogPeakGE        = []

for i in range(6):
    IDs.append(fullOBJs[i][0])
    densities.append(fullOBJs[i][1]/maxesForVarsOnly[0])
    costs.append(fullOBJs[i][2])
    LogXSgamABe6 .append(np.log10(fullOBJs[i][3])/maxesForVarsOnly[1])
    LogXSgamABe7.append(np.log10(fullOBJs[i][4])/maxesForVarsOnly[2])
    LogXSgamPROe_2.append(np.log10(fullOBJs[i][5])/maxesForVarsOnly[3])
    LogXSgamPROe_1.append(np.log10(fullOBJs[i][6])/maxesForVarsOnly[4])
    LogXSgamPROe4.append(np.log10(fullOBJs[i][7])/maxesForVarsOnly[5])
    LogXSgamPROe5.append(np.log10(fullOBJs[i][8])/maxesForVarsOnly[6])
    LogXSgamPROe7.append(np.log10(fullOBJs[i][10])/maxesForVarsOnly[7])
    LogXSalphaPROe_2.append(np.log10(fullOBJs[i][11])/maxesForVarsOnly[8])
    LogXSalphaPROe_1.append(np.log10(fullOBJs[i][12])/maxesForVarsOnly[9])
    LogXSalphaPROe4.append(np.log10(fullOBJs[i][13])/maxesForVarsOnly[10])
    LogXSalphaPROe5.append(np.log10(fullOBJs[i][14])/maxesForVarsOnly[11])
    LogXSalphaPROe6.append(np.log10(fullOBJs[i][15])/maxesForVarsOnly[12])
    LogXSalphaPROe7.append(np.log10(fullOBJs[i][16])/maxesForVarsOnly[13])
    LogXSelasticE_2.append(np.log10(fullOBJs[i][17])/maxesForVarsOnly[14])
    LogXSelasticE_1.append(np.log10(fullOBJs[i][18])/maxesForVarsOnly[15])
    LogXSelasticE4.append(np.log10(fullOBJs[i][19])/maxesForVarsOnly[16])
    LogXSelasticE5.append(np.log10(fullOBJs[i][20])/maxesForVarsOnly[17])
    LogXSelasticE6.append(np.log10(fullOBJs[i][21])/maxesForVarsOnly[18])
    LogXSelasticE7.append(np.log10(fullOBJs[i][22])/maxesForVarsOnly[19])
    LogPeakElastic.append(np.log10(fullOBJs[i][23])/maxesForVarsOnly[20])
    LogPeakAlpha.append(np.log10(fullOBJs[i][24])/maxesForVarsOnly[21])
    LogPeakGA.append(np.log10(fullOBJs[i][25])/maxesForVarsOnly[22])
    LogPeakGE.append(np.log10(fullOBJs[i][26])/maxesForVarsOnly[23])

print(IDs,'\n',densities,'\n',LogPeakGE)

###### Define material option codes #####
magicNum  = 10                     # number of sections the radial domain is divided into
costLimit = 3000000                # USD, acceptable limit of material cost
massLimit = 200000                 # kg, acceptable limit of material mass

print('MagicNum =',magicNum)

# Define File Name
fileName     = 'data.pkl'
maxValsPath  = 'maxVals.csv'
maxVals      = np.genfromtxt(maxValsPath, delimiter=',', skip_header=1)

lwbd = 0     # integer lower bound for all densityVar variables

####### Initialize GEKKO Model ####

m = GEKKO(remote=True)
loaded_model = joblib.load(fileName)

Model = gekko_sk(loaded_model,m)

L0guess  = 1
L1guess  = 2
L2guess  = 1
L3guess  = 1
L4guess  = 3
L5guess  = 1
L6guess  = 2
L7guess  = 1
L8guess  = 3
L9guess  = 1

### Layer 0 ###
# LAYER 0 VARIABLES
densityVar0           = m.Var(name='densityVar0',value=L0guess, lb=lwbd, ub=5, integer=True)
densityVarY0          = m.Var(name='densityVarY0')
densityNaveVarY0      = m.Intermediate(densityVarY0)
gamABe6VarY0          = m.Var(name='gamABe6VarY0')
gamABe7NaveVarY0      = m.Var(name='gamABe7NaveVarY0')
gamPROe_2NaveVarY0    = m.Var(name='gamPROe_2NaveVarY0')
gamPROe_1NaveVarY0    = m.Var(name='gamPROe_1NaveVarY0')
gamPROe4NaveVarY0     = m.Var(name='gamPROe4NaveVarY0')
gamPROe5NaveVarY0     = m.Var(name='gamPROe5NaveVarY0')
gamPROe7NaveVarY0     = m.Var(name='gamPROe7NaveVarY0')
alphaPROe_2NaveVarY0  = m.Var(name='alphaPROe_2NaveVarY0')
alphaPROe_1NaveVarY0  = m.Var(name='alphaPROe_1NaveVarY0')
alphaPROe4NaveVarY0   = m.Var(name='alphaPROe4NaveVarY0')
alphaPROe5NaveVarY0   = m.Var(name='alphaPROe5NaveVarY0')
alphaPROe6NaveVarY0   = m.Var(name='alphaPROe6NaveVarY0')
alphaPROe7NaveVarY0   = m.Var(name='alphaPROe7NaveVarY0')
elasticE_2NaveVarY0   = m.Var(name='elasticE_2NaveVarY0')
elasticE_1NaveVarY0   = m.Var(name='elasticE_1NaveVarY0')
elasticE4NaveVarY0    = m.Var(name='elasticE4NaveVarY0')
elasticE5NaveVarY0    = m.Var(name='elasticE5NaveVarY0')
elasticE6NaveVarY0    = m.Var(name='elasticE6NaveVarY0')
elasticE7NaveVarY0    = m.Var(name='elasticE7NaveVarY0')
PeakElasticNaveVarY0  = m.Var(name='PeakElasticNaveVarY0')
PeakAlphaNaveVarY0    = m.Var(name='PeakAlphaNaveVarY0')
PeakGANaveVarY0       = m.Var(name='PeakGANaveVarY0')
PeakGENaveVarY0       = m.Var(name='PeakGENaveVarY0')
costVarY0             = m.Var(name='PriceVarY0')

var0List = [densityVar0,densityVarY0,densityNaveVarY0,gamABe6VarY0,gamABe7NaveVarY0,gamPROe_2NaveVarY0,gamPROe_1NaveVarY0,\
            gamPROe4NaveVarY0,gamPROe5NaveVarY0,gamPROe7NaveVarY0,alphaPROe_2NaveVarY0,alphaPROe_1NaveVarY0,\
            alphaPROe4NaveVarY0,alphaPROe5NaveVarY0,alphaPROe6NaveVarY0,alphaPROe7NaveVarY0,elasticE_2NaveVarY0,\
            elasticE_1NaveVarY0,elasticE4NaveVarY0,elasticE5NaveVarY0,elasticE6NaveVarY0,elasticE7NaveVarY0,\
            PeakElasticNaveVarY0,PeakAlphaNaveVarY0,PeakGANaveVarY0,PeakGENaveVarY0,costVarY0]

# Layer 0 Splines
m.cspline(densityVar0, densityVarY0, IDs, densities)
m.cspline(densityVar0, gamABe6VarY0, IDs, LogXSgamABe6)
m.cspline(densityVar0, gamABe7NaveVarY0, IDs, LogXSgamABe7)
m.cspline(densityVar0, gamPROe_2NaveVarY0, IDs, LogXSgamPROe_2)
m.cspline(densityVar0, gamPROe_1NaveVarY0, IDs, LogXSgamPROe_1)
m.cspline(densityVar0, gamPROe4NaveVarY0, IDs, LogXSgamPROe4)
m.cspline(densityVar0, gamPROe5NaveVarY0, IDs, LogXSgamPROe5)
### This one was skipped because it's not in the model ###
m.cspline(densityVar0, gamPROe7NaveVarY0, IDs, LogXSgamPROe7)
m.cspline(densityVar0, alphaPROe_2NaveVarY0, IDs, LogXSalphaPROe_2)
m.cspline(densityVar0, alphaPROe_1NaveVarY0, IDs, LogXSalphaPROe_1)
m.cspline(densityVar0, alphaPROe4NaveVarY0, IDs, LogXSalphaPROe4)
m.cspline(densityVar0, alphaPROe5NaveVarY0, IDs, LogXSalphaPROe5)
m.cspline(densityVar0, alphaPROe6NaveVarY0, IDs, LogXSalphaPROe6)
m.cspline(densityVar0, alphaPROe7NaveVarY0, IDs, LogXSalphaPROe7)
m.cspline(densityVar0, elasticE_2NaveVarY0, IDs, LogXSelasticE_2)
m.cspline(densityVar0, elasticE_1NaveVarY0, IDs, LogXSelasticE_1)
m.cspline(densityVar0, elasticE4NaveVarY0, IDs, LogXSelasticE4)
m.cspline(densityVar0, elasticE5NaveVarY0, IDs, LogXSelasticE5)
m.cspline(densityVar0, elasticE6NaveVarY0, IDs, LogXSelasticE6)
m.cspline(densityVar0, elasticE7NaveVarY0, IDs, LogXSelasticE7)
m.cspline(densityVar0, PeakElasticNaveVarY0, IDs, LogPeakElastic)
m.cspline(densityVar0, PeakAlphaNaveVarY0, IDs, LogPeakAlpha)
m.cspline(densityVar0, PeakGANaveVarY0, IDs, LogPeakGA)
m.cspline(densityVar0, PeakGENaveVarY0, IDs, LogPeakGE)
m.cspline(densityVar0, costVarY0, IDs, costs)

### Layer 1 ###
# LAYER 1 VARIABLES
densityVar1           = m.Var(name='densityVar1',value=L1guess, lb=lwbd, ub=5, integer=True)
densityVarY1          = m.Var(name='densityVarY1')
densityNaveVarY1      = m.Intermediate(densityVarY1)
gamABe6VarY1          = m.Var(name='gamABe6VarY1')
gamABe7NaveVarY1      = m.Var(name='gamABe7NaveVarY1')
gamPROe_2NaveVarY1    = m.Var(name='gamPROe_2NaveVarY1')
gamPROe_1NaveVarY1    = m.Var(name='gamPROe_1NaveVarY1')
gamPROe4NaveVarY1     = m.Var(name='gamPROe4NaveVarY1')
gamPROe5NaveVarY1     = m.Var(name='gamPROe5NaveVarY1')
gamPROe7NaveVarY1     = m.Var(name='gamPROe7NaveVarY1')
alphaPROe_2NaveVarY1  = m.Var(name='alphaPROe_2NaveVarY1')
alphaPROe_1NaveVarY1  = m.Var(name='alphaPROe_1NaveVarY1')
alphaPROe4NaveVarY1   = m.Var(name='alphaPROe4NaveVarY1')
alphaPROe5NaveVarY1   = m.Var(name='alphaPROe5NaveVarY1')
alphaPROe6NaveVarY1   = m.Var(name='alphaPROe6NaveVarY1')
alphaPROe7NaveVarY1   = m.Var(name='alphaPROe7NaveVarY1')
elasticE_2NaveVarY1   = m.Var(name='elasticE_2NaveVarY1')
elasticE_1NaveVarY1   = m.Var(name='elasticE_1NaveVarY1')
elasticE4NaveVarY1    = m.Var(name='elasticE4NaveVarY1')
elasticE5NaveVarY1    = m.Var(name='elasticE5NaveVarY1')
elasticE6NaveVarY1    = m.Var(name='elasticE6NaveVarY1')
elasticE7NaveVarY1    = m.Var(name='elasticE7NaveVarY1')
PeakElasticNaveVarY1  = m.Var(name='PeakElasticNaveVarY1')
PeakAlphaNaveVarY1    = m.Var(name='PeakAlphaNaveVarY1')
PeakGANaveVarY1       = m.Var(name='PeakGANaveVarY1')
PeakGENaveVarY1       = m.Var(name='PeakGENaveVarY1')
costVarY1             = m.Var(name='PriceVarY1')

var1List = [densityVar1,densityVarY1,densityNaveVarY1,gamABe6VarY1,gamABe7NaveVarY1,gamPROe_2NaveVarY1,gamPROe_1NaveVarY1,\
            gamPROe4NaveVarY1,gamPROe5NaveVarY1,gamPROe7NaveVarY1,alphaPROe_2NaveVarY1,alphaPROe_1NaveVarY1,\
            alphaPROe4NaveVarY1,alphaPROe5NaveVarY1,alphaPROe6NaveVarY1,alphaPROe7NaveVarY1,elasticE_2NaveVarY1,\
            elasticE_1NaveVarY1,elasticE4NaveVarY1,elasticE5NaveVarY1,elasticE6NaveVarY1,elasticE7NaveVarY1,\
            PeakElasticNaveVarY1,PeakAlphaNaveVarY1,PeakGANaveVarY1,PeakGENaveVarY1,costVarY1]

# Layer 1 Splines
m.cspline(densityVar1, densityVarY1, IDs, densities)
m.cspline(densityVar1, gamABe6VarY1, IDs, LogXSgamABe6)
m.cspline(densityVar1, gamABe7NaveVarY1, IDs, LogXSgamABe7)
m.cspline(densityVar1, gamPROe_2NaveVarY1, IDs, LogXSgamPROe_2)
m.cspline(densityVar1, gamPROe_1NaveVarY1, IDs, LogXSgamPROe_1)
m.cspline(densityVar1, gamPROe4NaveVarY1, IDs, LogXSgamPROe4)
m.cspline(densityVar1, gamPROe5NaveVarY1, IDs, LogXSgamPROe5)
### This one was skipped because it's not in the model ###
m.cspline(densityVar1, gamPROe7NaveVarY1, IDs, LogXSgamPROe7)
m.cspline(densityVar1, alphaPROe_2NaveVarY1, IDs, LogXSalphaPROe_2)
m.cspline(densityVar1, alphaPROe_1NaveVarY1, IDs, LogXSalphaPROe_1)
m.cspline(densityVar1, alphaPROe4NaveVarY1, IDs, LogXSalphaPROe4)
m.cspline(densityVar1, alphaPROe5NaveVarY1, IDs, LogXSalphaPROe5)
m.cspline(densityVar1, alphaPROe6NaveVarY1, IDs, LogXSalphaPROe6)
m.cspline(densityVar1, alphaPROe7NaveVarY1, IDs, LogXSalphaPROe7)
m.cspline(densityVar1, elasticE_2NaveVarY1, IDs, LogXSelasticE_2)
m.cspline(densityVar1, elasticE_1NaveVarY1, IDs, LogXSelasticE_1)
m.cspline(densityVar1, elasticE4NaveVarY1, IDs, LogXSelasticE4)
m.cspline(densityVar1, elasticE5NaveVarY1, IDs, LogXSelasticE5)
m.cspline(densityVar1, elasticE6NaveVarY1, IDs, LogXSelasticE6)
m.cspline(densityVar1, elasticE7NaveVarY1, IDs, LogXSelasticE7)
m.cspline(densityVar1, PeakElasticNaveVarY1, IDs, LogPeakElastic)
m.cspline(densityVar1, PeakAlphaNaveVarY1, IDs, LogPeakAlpha)
m.cspline(densityVar1, PeakGANaveVarY1, IDs, LogPeakGA)
m.cspline(densityVar1, PeakGENaveVarY1, IDs, LogPeakGE)
m.cspline(densityVar1, costVarY1, IDs, costs)

### Layer 2 ###
# LAYER 2 VARIABLES m.Param(value=L2guess,name='densityVar2') #
densityVar2           = m.Var(name='densityVar2',value=L2guess, lb=lwbd, ub=5, integer=True)
densityVarY2          = m.Var(name='densityVarY2')
densityNaveVarY2      = m.Intermediate(densityVarY2)
gamABe6VarY2          = m.Var(name='gamABe6VarY2')
gamABe7NaveVarY2      = m.Var(name='gamABe7NaveVarY2')
gamPROe_2NaveVarY2    = m.Var(name='gamPROe_2NaveVarY2')
gamPROe_1NaveVarY2    = m.Var(name='gamPROe_1NaveVarY2')
gamPROe4NaveVarY2     = m.Var(name='gamPROe4NaveVarY2')
gamPROe5NaveVarY2     = m.Var(name='gamPROe5NaveVarY2')
gamPROe7NaveVarY2     = m.Var(name='gamPROe7NaveVarY2')
alphaPROe_2NaveVarY2  = m.Var(name='alphaPROe_2NaveVarY2')
alphaPROe_1NaveVarY2  = m.Var(name='alphaPROe_1NaveVarY2')
alphaPROe4NaveVarY2   = m.Var(name='alphaPROe4NaveVarY2')
alphaPROe5NaveVarY2   = m.Var(name='alphaPROe5NaveVarY2')
alphaPROe6NaveVarY2   = m.Var(name='alphaPROe6NaveVarY2')
alphaPROe7NaveVarY2   = m.Var(name='alphaPROe7NaveVarY2')
elasticE_2NaveVarY2   = m.Var(name='elasticE_2NaveVarY2')
elasticE_1NaveVarY2   = m.Var(name='elasticE_1NaveVarY2')
elasticE4NaveVarY2    = m.Var(name='elasticE4NaveVarY2')
elasticE5NaveVarY2    = m.Var(name='elasticE5NaveVarY2')
elasticE6NaveVarY2    = m.Var(name='elasticE6NaveVarY2')
elasticE7NaveVarY2    = m.Var(name='elasticE7NaveVarY2')
PeakElasticNaveVarY2  = m.Var(name='PeakElasticNaveVarY2')
PeakAlphaNaveVarY2    = m.Var(name='PeakAlphaNaveVarY2')
PeakGANaveVarY2       = m.Var(name='PeakGANaveVarY2')
PeakGENaveVarY2       = m.Var(name='PeakGENaveVarY2')
costVarY2             = m.Var(name='PriceVarY2')

var2List = [densityVar2,densityVarY2,densityNaveVarY2,gamABe6VarY2,gamABe7NaveVarY2,gamPROe_2NaveVarY2,gamPROe_1NaveVarY2,\
            gamPROe4NaveVarY2,gamPROe5NaveVarY2,gamPROe7NaveVarY2,alphaPROe_2NaveVarY2,alphaPROe_1NaveVarY2,\
            alphaPROe4NaveVarY2,alphaPROe5NaveVarY2,alphaPROe6NaveVarY2,alphaPROe7NaveVarY2,elasticE_2NaveVarY2,\
            elasticE_1NaveVarY2,elasticE4NaveVarY2,elasticE5NaveVarY2,elasticE6NaveVarY2,elasticE7NaveVarY2,\
            PeakElasticNaveVarY2,PeakAlphaNaveVarY2,PeakGANaveVarY2,PeakGENaveVarY2,costVarY2]

# Layer 2 Splines
m.cspline(densityVar2, densityVarY2, IDs, densities)
m.cspline(densityVar2, gamABe6VarY2, IDs, LogXSgamABe6)
m.cspline(densityVar2, gamABe7NaveVarY2, IDs, LogXSgamABe7)
m.cspline(densityVar2, gamPROe_2NaveVarY2, IDs, LogXSgamPROe_2)
m.cspline(densityVar2, gamPROe_1NaveVarY2, IDs, LogXSgamPROe_1)
m.cspline(densityVar2, gamPROe4NaveVarY2, IDs, LogXSgamPROe4)
m.cspline(densityVar2, gamPROe5NaveVarY2, IDs, LogXSgamPROe5)
### This one was skipped because it's not in the model ###
m.cspline(densityVar2, gamPROe7NaveVarY2, IDs, LogXSgamPROe7)
m.cspline(densityVar2, alphaPROe_2NaveVarY2, IDs, LogXSalphaPROe_2)
m.cspline(densityVar2, alphaPROe_1NaveVarY2, IDs, LogXSalphaPROe_1)
m.cspline(densityVar2, alphaPROe4NaveVarY2, IDs, LogXSalphaPROe4)
m.cspline(densityVar2, alphaPROe5NaveVarY2, IDs, LogXSalphaPROe5)
m.cspline(densityVar2, alphaPROe6NaveVarY2, IDs, LogXSalphaPROe6)
m.cspline(densityVar2, alphaPROe7NaveVarY2, IDs, LogXSalphaPROe7)
m.cspline(densityVar2, elasticE_2NaveVarY2, IDs, LogXSelasticE_2)
m.cspline(densityVar2, elasticE_1NaveVarY2, IDs, LogXSelasticE_1)
m.cspline(densityVar2, elasticE4NaveVarY2, IDs, LogXSelasticE4)
m.cspline(densityVar2, elasticE5NaveVarY2, IDs, LogXSelasticE5)
m.cspline(densityVar2, elasticE6NaveVarY2, IDs, LogXSelasticE6)
m.cspline(densityVar2, elasticE7NaveVarY2, IDs, LogXSelasticE7)
m.cspline(densityVar2, PeakElasticNaveVarY2, IDs, LogPeakElastic)
m.cspline(densityVar2, PeakAlphaNaveVarY2, IDs, LogPeakAlpha)
m.cspline(densityVar2, PeakGANaveVarY2, IDs, LogPeakGA)
m.cspline(densityVar2, PeakGENaveVarY2, IDs, LogPeakGE)
m.cspline(densityVar2, costVarY2, IDs, costs)

### Layer 3 ###
# LAYER 3 VARIABLES
densityVar3           = m.Var(name='densityVar3',value=L3guess, lb=lwbd, ub=5, integer=True)
densityVarY3          = m.Var(name='densityVarY3')
densityNaveVarY3      = m.Intermediate(densityVarY3)
gamABe6VarY3          = m.Var(name='gamABe6VarY3')
gamABe7NaveVarY3      = m.Var(name='gamABe7NaveVarY3')
gamPROe_2NaveVarY3    = m.Var(name='gamPROe_2NaveVarY3')
gamPROe_1NaveVarY3    = m.Var(name='gamPROe_1NaveVarY3')
gamPROe4NaveVarY3     = m.Var(name='gamPROe4NaveVarY3')
gamPROe5NaveVarY3     = m.Var(name='gamPROe5NaveVarY3')
gamPROe7NaveVarY3     = m.Var(name='gamPROe7NaveVarY3')
alphaPROe_2NaveVarY3  = m.Var(name='alphaPROe_2NaveVarY3')
alphaPROe_1NaveVarY3  = m.Var(name='alphaPROe_1NaveVarY3')
alphaPROe4NaveVarY3   = m.Var(name='alphaPROe4NaveVarY3')
alphaPROe5NaveVarY3   = m.Var(name='alphaPROe5NaveVarY3')
alphaPROe6NaveVarY3   = m.Var(name='alphaPROe6NaveVarY3')
alphaPROe7NaveVarY3   = m.Var(name='alphaPROe7NaveVarY3')
elasticE_2NaveVarY3   = m.Var(name='elasticE_2NaveVarY3')
elasticE_1NaveVarY3   = m.Var(name='elasticE_1NaveVarY3')
elasticE4NaveVarY3    = m.Var(name='elasticE4NaveVarY3')
elasticE5NaveVarY3    = m.Var(name='elasticE5NaveVarY3')
elasticE6NaveVarY3    = m.Var(name='elasticE6NaveVarY3')
elasticE7NaveVarY3    = m.Var(name='elasticE7NaveVarY3')
PeakElasticNaveVarY3  = m.Var(name='PeakElasticNaveVarY3')
PeakAlphaNaveVarY3    = m.Var(name='PeakAlphaNaveVarY3')
PeakGANaveVarY3       = m.Var(name='PeakGANaveVarY3')
PeakGENaveVarY3       = m.Var(name='PeakGENaveVarY3')
costVarY3             = m.Var(name='PriceVarY3')

var3List = [densityVar3,densityVarY3,densityNaveVarY3,gamABe6VarY3,gamABe7NaveVarY3,gamPROe_2NaveVarY3,gamPROe_1NaveVarY3,\
            gamPROe4NaveVarY3,gamPROe5NaveVarY3,gamPROe7NaveVarY3,alphaPROe_2NaveVarY3,alphaPROe_1NaveVarY3,\
            alphaPROe4NaveVarY3,alphaPROe5NaveVarY3,alphaPROe6NaveVarY3,alphaPROe7NaveVarY3,elasticE_2NaveVarY3,\
            elasticE_1NaveVarY3,elasticE4NaveVarY3,elasticE5NaveVarY3,elasticE6NaveVarY3,elasticE7NaveVarY3,\
            PeakElasticNaveVarY3,PeakAlphaNaveVarY3,PeakGANaveVarY3,PeakGENaveVarY3,costVarY3]

# Layer 3 Splines
m.cspline(densityVar3, densityVarY3, IDs, densities)
m.cspline(densityVar3, gamABe6VarY3, IDs, LogXSgamABe6)
m.cspline(densityVar3, gamABe7NaveVarY3, IDs, LogXSgamABe7)
m.cspline(densityVar3, gamPROe_2NaveVarY3, IDs, LogXSgamPROe_2)
m.cspline(densityVar3, gamPROe_1NaveVarY3, IDs, LogXSgamPROe_1)
m.cspline(densityVar3, gamPROe4NaveVarY3, IDs, LogXSgamPROe4)
m.cspline(densityVar3, gamPROe5NaveVarY3, IDs, LogXSgamPROe5)
### This one was skipped because it's not in the model ###
m.cspline(densityVar3, gamPROe7NaveVarY3, IDs, LogXSgamPROe7)
m.cspline(densityVar3, alphaPROe_2NaveVarY3, IDs, LogXSalphaPROe_2)
m.cspline(densityVar3, alphaPROe_1NaveVarY3, IDs, LogXSalphaPROe_1)
m.cspline(densityVar3, alphaPROe4NaveVarY3, IDs, LogXSalphaPROe4)
m.cspline(densityVar3, alphaPROe5NaveVarY3, IDs, LogXSalphaPROe5)
m.cspline(densityVar3, alphaPROe6NaveVarY3, IDs, LogXSalphaPROe6)
m.cspline(densityVar3, alphaPROe7NaveVarY3, IDs, LogXSalphaPROe7)
m.cspline(densityVar3, elasticE_2NaveVarY3, IDs, LogXSelasticE_2)
m.cspline(densityVar3, elasticE_1NaveVarY3, IDs, LogXSelasticE_1)
m.cspline(densityVar3, elasticE4NaveVarY3, IDs, LogXSelasticE4)
m.cspline(densityVar3, elasticE5NaveVarY3, IDs, LogXSelasticE5)
m.cspline(densityVar3, elasticE6NaveVarY3, IDs, LogXSelasticE6)
m.cspline(densityVar3, elasticE7NaveVarY3, IDs, LogXSelasticE7)
m.cspline(densityVar3, PeakElasticNaveVarY3, IDs, LogPeakElastic)
m.cspline(densityVar3, PeakAlphaNaveVarY3, IDs, LogPeakAlpha)
m.cspline(densityVar3, PeakGANaveVarY3, IDs, LogPeakGA)
m.cspline(densityVar3, PeakGENaveVarY3, IDs, LogPeakGE)
m.cspline(densityVar3, costVarY3, IDs, costs)

### Layer 4 ###
# LAYER 4 VARIABLES
densityVar4           = m.Var(name='densityVar4',value=L4guess, lb=lwbd, ub=5, integer=True)
densityVarY4          = m.Var(name='densityVarY4')
densityNaveVarY4      = m.Intermediate(densityVarY4)
gamABe6VarY4          = m.Var(name='gamABe6VarY4')
gamABe7NaveVarY4      = m.Var(name='gamABe7NaveVarY4')
gamPROe_2NaveVarY4    = m.Var(name='gamPROe_2NaveVarY4')
gamPROe_1NaveVarY4    = m.Var(name='gamPROe_1NaveVarY4')
gamPROe4NaveVarY4     = m.Var(name='gamPROe4NaveVarY4')
gamPROe5NaveVarY4     = m.Var(name='gamPROe5NaveVarY4')
gamPROe7NaveVarY4     = m.Var(name='gamPROe7NaveVarY4')
alphaPROe_2NaveVarY4  = m.Var(name='alphaPROe_2NaveVarY4')
alphaPROe_1NaveVarY4  = m.Var(name='alphaPROe_1NaveVarY4')
alphaPROe4NaveVarY4   = m.Var(name='alphaPROe4NaveVarY4')
alphaPROe5NaveVarY4   = m.Var(name='alphaPROe5NaveVarY4')
alphaPROe6NaveVarY4   = m.Var(name='alphaPROe6NaveVarY4')
alphaPROe7NaveVarY4   = m.Var(name='alphaPROe7NaveVarY4')
elasticE_2NaveVarY4   = m.Var(name='elasticE_2NaveVarY4')
elasticE_1NaveVarY4   = m.Var(name='elasticE_1NaveVarY4')
elasticE4NaveVarY4    = m.Var(name='elasticE4NaveVarY4')
elasticE5NaveVarY4    = m.Var(name='elasticE5NaveVarY4')
elasticE6NaveVarY4    = m.Var(name='elasticE6NaveVarY4')
elasticE7NaveVarY4    = m.Var(name='elasticE7NaveVarY4')
PeakElasticNaveVarY4  = m.Var(name='PeakElasticNaveVarY4')
PeakAlphaNaveVarY4    = m.Var(name='PeakAlphaNaveVarY4')
PeakGANaveVarY4       = m.Var(name='PeakGANaveVarY4')
PeakGENaveVarY4       = m.Var(name='PeakGENaveVarY4')
costVarY4             = m.Var(name='PriceVarY4')

var4List = [densityVar4,densityVarY4,densityNaveVarY4,gamABe6VarY4,gamABe7NaveVarY4,gamPROe_2NaveVarY4,gamPROe_1NaveVarY4,\
            gamPROe4NaveVarY4,gamPROe5NaveVarY4,gamPROe7NaveVarY4,alphaPROe_2NaveVarY4,alphaPROe_1NaveVarY4,\
            alphaPROe4NaveVarY4,alphaPROe5NaveVarY4,alphaPROe6NaveVarY4,alphaPROe7NaveVarY4,elasticE_2NaveVarY4,\
            elasticE_1NaveVarY4,elasticE4NaveVarY4,elasticE5NaveVarY4,elasticE6NaveVarY4,elasticE7NaveVarY4,\
            PeakElasticNaveVarY4,PeakAlphaNaveVarY4,PeakGANaveVarY4,PeakGENaveVarY4,costVarY4]

# Layer 4 Splines
m.cspline(densityVar4, densityVarY4, IDs, densities)
m.cspline(densityVar4, gamABe6VarY4, IDs, LogXSgamABe6)
m.cspline(densityVar4, gamABe7NaveVarY4, IDs, LogXSgamABe7)
m.cspline(densityVar4, gamPROe_2NaveVarY4, IDs, LogXSgamPROe_2)
m.cspline(densityVar4, gamPROe_1NaveVarY4, IDs, LogXSgamPROe_1)
m.cspline(densityVar4, gamPROe4NaveVarY4, IDs, LogXSgamPROe4)
m.cspline(densityVar4, gamPROe5NaveVarY4, IDs, LogXSgamPROe5)
### This one was skipped because it's not in the model ###
m.cspline(densityVar4, gamPROe7NaveVarY4, IDs, LogXSgamPROe7)
m.cspline(densityVar4, alphaPROe_2NaveVarY4, IDs, LogXSalphaPROe_2)
m.cspline(densityVar4, alphaPROe_1NaveVarY4, IDs, LogXSalphaPROe_1)
m.cspline(densityVar4, alphaPROe4NaveVarY4, IDs, LogXSalphaPROe4)
m.cspline(densityVar4, alphaPROe5NaveVarY4, IDs, LogXSalphaPROe5)
m.cspline(densityVar4, alphaPROe6NaveVarY4, IDs, LogXSalphaPROe6)
m.cspline(densityVar4, alphaPROe7NaveVarY4, IDs, LogXSalphaPROe7)
m.cspline(densityVar4, elasticE_2NaveVarY4, IDs, LogXSelasticE_2)
m.cspline(densityVar4, elasticE_1NaveVarY4, IDs, LogXSelasticE_1)
m.cspline(densityVar4, elasticE4NaveVarY4, IDs, LogXSelasticE4)
m.cspline(densityVar4, elasticE5NaveVarY4, IDs, LogXSelasticE5)
m.cspline(densityVar4, elasticE6NaveVarY4, IDs, LogXSelasticE6)
m.cspline(densityVar4, elasticE7NaveVarY4, IDs, LogXSelasticE7)
m.cspline(densityVar4, PeakElasticNaveVarY4, IDs, LogPeakElastic)
m.cspline(densityVar4, PeakAlphaNaveVarY4, IDs, LogPeakAlpha)
m.cspline(densityVar4, PeakGANaveVarY4, IDs, LogPeakGA)
m.cspline(densityVar4, PeakGENaveVarY4, IDs, LogPeakGE)
m.cspline(densityVar4, costVarY4, IDs, costs)

### Layer 5 ###
# LAYER 5 VARIABLES
densityVar5           = m.Var(name='densityVar5',value=L5guess, lb=lwbd, ub=5, integer=True)
densityVarY5          = m.Var(name='densityVarY5')
densityNaveVarY5      = m.Intermediate(densityVarY5)
gamABe6VarY5          = m.Var(name='gamABe6VarY5')
gamABe7NaveVarY5      = m.Var(name='gamABe7NaveVarY5')
gamPROe_2NaveVarY5    = m.Var(name='gamPROe_2NaveVarY5')
gamPROe_1NaveVarY5    = m.Var(name='gamPROe_1NaveVarY5')
gamPROe4NaveVarY5     = m.Var(name='gamPROe4NaveVarY5')
gamPROe5NaveVarY5     = m.Var(name='gamPROe5NaveVarY5')
gamPROe7NaveVarY5     = m.Var(name='gamPROe7NaveVarY5')
alphaPROe_2NaveVarY5  = m.Var(name='alphaPROe_2NaveVarY5')
alphaPROe_1NaveVarY5  = m.Var(name='alphaPROe_1NaveVarY5')
alphaPROe4NaveVarY5   = m.Var(name='alphaPROe4NaveVarY5')
alphaPROe5NaveVarY5   = m.Var(name='alphaPROe5NaveVarY5')
alphaPROe6NaveVarY5   = m.Var(name='alphaPROe6NaveVarY5')
alphaPROe7NaveVarY5   = m.Var(name='alphaPROe7NaveVarY5')
elasticE_2NaveVarY5   = m.Var(name='elasticE_2NaveVarY5')
elasticE_1NaveVarY5   = m.Var(name='elasticE_1NaveVarY5')
elasticE4NaveVarY5    = m.Var(name='elasticE4NaveVarY5')
elasticE5NaveVarY5    = m.Var(name='elasticE5NaveVarY5')
elasticE6NaveVarY5    = m.Var(name='elasticE6NaveVarY5')
elasticE7NaveVarY5    = m.Var(name='elasticE7NaveVarY5')
PeakElasticNaveVarY5  = m.Var(name='PeakElasticNaveVarY5')
PeakAlphaNaveVarY5    = m.Var(name='PeakAlphaNaveVarY5')
PeakGANaveVarY5       = m.Var(name='PeakGANaveVarY5')
PeakGENaveVarY5       = m.Var(name='PeakGENaveVarY5')
costVarY5             = m.Var(name='PriceVarY5')

var5List = [densityVar5,densityVarY5,densityNaveVarY5,gamABe6VarY5,gamABe7NaveVarY5,gamPROe_2NaveVarY5,gamPROe_1NaveVarY5,\
            gamPROe4NaveVarY5,gamPROe5NaveVarY5,gamPROe7NaveVarY5,alphaPROe_2NaveVarY5,alphaPROe_1NaveVarY5,\
            alphaPROe4NaveVarY5,alphaPROe5NaveVarY5,alphaPROe6NaveVarY5,alphaPROe7NaveVarY5,elasticE_2NaveVarY5,\
            elasticE_1NaveVarY5,elasticE4NaveVarY5,elasticE5NaveVarY5,elasticE6NaveVarY5,elasticE7NaveVarY5,\
            PeakElasticNaveVarY5,PeakAlphaNaveVarY5,PeakGANaveVarY5,PeakGENaveVarY5,costVarY5]

# Layer 5 Splines
m.cspline(densityVar5, densityVarY5, IDs, densities)
m.cspline(densityVar5, gamABe6VarY5, IDs, LogXSgamABe6)
m.cspline(densityVar5, gamABe7NaveVarY5, IDs, LogXSgamABe7)
m.cspline(densityVar5, gamPROe_2NaveVarY5, IDs, LogXSgamPROe_2)
m.cspline(densityVar5, gamPROe_1NaveVarY5, IDs, LogXSgamPROe_1)
m.cspline(densityVar5, gamPROe4NaveVarY5, IDs, LogXSgamPROe4)
m.cspline(densityVar5, gamPROe5NaveVarY5, IDs, LogXSgamPROe5)
### This one was skipped because it's not in the model ###
m.cspline(densityVar5, gamPROe7NaveVarY5, IDs, LogXSgamPROe7)
m.cspline(densityVar5, alphaPROe_2NaveVarY5, IDs, LogXSalphaPROe_2)
m.cspline(densityVar5, alphaPROe_1NaveVarY5, IDs, LogXSalphaPROe_1)
m.cspline(densityVar5, alphaPROe4NaveVarY5, IDs, LogXSalphaPROe4)
m.cspline(densityVar5, alphaPROe5NaveVarY5, IDs, LogXSalphaPROe5)
m.cspline(densityVar5, alphaPROe6NaveVarY5, IDs, LogXSalphaPROe6)
m.cspline(densityVar5, alphaPROe7NaveVarY5, IDs, LogXSalphaPROe7)
m.cspline(densityVar5, elasticE_2NaveVarY5, IDs, LogXSelasticE_2)
m.cspline(densityVar5, elasticE_1NaveVarY5, IDs, LogXSelasticE_1)
m.cspline(densityVar5, elasticE4NaveVarY5, IDs, LogXSelasticE4)
m.cspline(densityVar5, elasticE5NaveVarY5, IDs, LogXSelasticE5)
m.cspline(densityVar5, elasticE6NaveVarY5, IDs, LogXSelasticE6)
m.cspline(densityVar5, elasticE7NaveVarY5, IDs, LogXSelasticE7)
m.cspline(densityVar5, PeakElasticNaveVarY5, IDs, LogPeakElastic)
m.cspline(densityVar5, PeakAlphaNaveVarY5, IDs, LogPeakAlpha)
m.cspline(densityVar5, PeakGANaveVarY5, IDs, LogPeakGA)
m.cspline(densityVar5, PeakGENaveVarY5, IDs, LogPeakGE)
m.cspline(densityVar5, costVarY5, IDs, costs)

# ### Layer 6 ###
# LAYER 6 VARIABLES
densityVar6           = m.Var(name='densityVar6',value=L6guess, lb=lwbd, ub=5, integer=True)
densityVarY6          = m.Var(name='densityVarY6')
densityNaveVarY6      = m.Intermediate(densityVarY6)
gamABe6VarY6          = m.Var(name='gamABe6VarY6')
gamABe7NaveVarY6      = m.Var(name='gamABe7NaveVarY6')
gamPROe_2NaveVarY6    = m.Var(name='gamPROe_2NaveVarY6')
gamPROe_1NaveVarY6    = m.Var(name='gamPROe_1NaveVarY6')
gamPROe4NaveVarY6     = m.Var(name='gamPROe4NaveVarY6')
gamPROe5NaveVarY6     = m.Var(name='gamPROe5NaveVarY6')
gamPROe7NaveVarY6     = m.Var(name='gamPROe7NaveVarY6')
alphaPROe_2NaveVarY6  = m.Var(name='alphaPROe_2NaveVarY6')
alphaPROe_1NaveVarY6  = m.Var(name='alphaPROe_1NaveVarY6')
alphaPROe4NaveVarY6   = m.Var(name='alphaPROe4NaveVarY6')
alphaPROe5NaveVarY6   = m.Var(name='alphaPROe5NaveVarY6')
alphaPROe6NaveVarY6   = m.Var(name='alphaPROe6NaveVarY6')
alphaPROe7NaveVarY6   = m.Var(name='alphaPROe7NaveVarY6')
elasticE_2NaveVarY6   = m.Var(name='elasticE_2NaveVarY6')
elasticE_1NaveVarY6   = m.Var(name='elasticE_1NaveVarY6')
elasticE4NaveVarY6    = m.Var(name='elasticE4NaveVarY6')
elasticE5NaveVarY6    = m.Var(name='elasticE5NaveVarY6')
elasticE6NaveVarY6    = m.Var(name='elasticE6NaveVarY6')
elasticE7NaveVarY6    = m.Var(name='elasticE7NaveVarY6')
PeakElasticNaveVarY6  = m.Var(name='PeakElasticNaveVarY6')
PeakAlphaNaveVarY6    = m.Var(name='PeakAlphaNaveVarY6')
PeakGANaveVarY6       = m.Var(name='PeakGANaveVarY6')
PeakGENaveVarY6       = m.Var(name='PeakGENaveVarY6')
costVarY6             = m.Var(name='PriceVarY6')

var6List = [densityVar6,densityVarY6,densityNaveVarY6,gamABe6VarY6,gamABe7NaveVarY6,gamPROe_2NaveVarY6,gamPROe_1NaveVarY6,\
            gamPROe4NaveVarY6,gamPROe5NaveVarY6,gamPROe7NaveVarY6,alphaPROe_2NaveVarY6,alphaPROe_1NaveVarY6,\
            alphaPROe4NaveVarY6,alphaPROe5NaveVarY6,alphaPROe6NaveVarY6,alphaPROe7NaveVarY6,elasticE_2NaveVarY6,\
            elasticE_1NaveVarY6,elasticE4NaveVarY6,elasticE5NaveVarY6,elasticE6NaveVarY6,elasticE7NaveVarY6,\
            PeakElasticNaveVarY6,PeakAlphaNaveVarY6,PeakGANaveVarY6,PeakGENaveVarY6,costVarY6]

# Layer 6 Splines
m.cspline(densityVar6, densityVarY6, IDs, densities)
m.cspline(densityVar6, gamABe6VarY6, IDs, LogXSgamABe6)
m.cspline(densityVar6, gamABe7NaveVarY6, IDs, LogXSgamABe7)
m.cspline(densityVar6, gamPROe_2NaveVarY6, IDs, LogXSgamPROe_2)
m.cspline(densityVar6, gamPROe_1NaveVarY6, IDs, LogXSgamPROe_1)
m.cspline(densityVar6, gamPROe4NaveVarY6, IDs, LogXSgamPROe4)
m.cspline(densityVar6, gamPROe5NaveVarY6, IDs, LogXSgamPROe5)
### This one was skipped because it's not in the model ###
m.cspline(densityVar6, gamPROe7NaveVarY6, IDs, LogXSgamPROe7)
m.cspline(densityVar6, alphaPROe_2NaveVarY6, IDs, LogXSalphaPROe_2)
m.cspline(densityVar6, alphaPROe_1NaveVarY6, IDs, LogXSalphaPROe_1)
m.cspline(densityVar6, alphaPROe4NaveVarY6, IDs, LogXSalphaPROe4)
m.cspline(densityVar6, alphaPROe5NaveVarY6, IDs, LogXSalphaPROe5)
m.cspline(densityVar6, alphaPROe6NaveVarY6, IDs, LogXSalphaPROe6)
m.cspline(densityVar6, alphaPROe7NaveVarY6, IDs, LogXSalphaPROe7)
m.cspline(densityVar6, elasticE_2NaveVarY6, IDs, LogXSelasticE_2)
m.cspline(densityVar6, elasticE_1NaveVarY6, IDs, LogXSelasticE_1)
m.cspline(densityVar6, elasticE4NaveVarY6, IDs, LogXSelasticE4)
m.cspline(densityVar6, elasticE5NaveVarY6, IDs, LogXSelasticE5)
m.cspline(densityVar6, elasticE6NaveVarY6, IDs, LogXSelasticE6)
m.cspline(densityVar6, elasticE7NaveVarY6, IDs, LogXSelasticE7)
m.cspline(densityVar6, PeakElasticNaveVarY6, IDs, LogPeakElastic)
m.cspline(densityVar6, PeakAlphaNaveVarY6, IDs, LogPeakAlpha)
m.cspline(densityVar6, PeakGANaveVarY6, IDs, LogPeakGA)
m.cspline(densityVar6, PeakGENaveVarY6, IDs, LogPeakGE)
m.cspline(densityVar6, costVarY6, IDs, costs)

### Layer 7 ###
# LAYER 7 VARIABLES
densityVar7           = m.Var(name='densityVar7',value=L7guess, lb=lwbd, ub=5, integer=True)
densityVarY7          = m.Var(name='densityVarY7')
densityNaveVarY7      = m.Intermediate(densityVarY7)
gamABe6VarY7          = m.Var(name='gamABe6VarY7')
gamABe7NaveVarY7      = m.Var(name='gamABe7NaveVarY7')
gamPROe_2NaveVarY7    = m.Var(name='gamPROe_2NaveVarY7')
gamPROe_1NaveVarY7    = m.Var(name='gamPROe_1NaveVarY7')
gamPROe4NaveVarY7     = m.Var(name='gamPROe4NaveVarY7')
gamPROe5NaveVarY7     = m.Var(name='gamPROe5NaveVarY7')
gamPROe7NaveVarY7     = m.Var(name='gamPROe7NaveVarY7')
alphaPROe_2NaveVarY7  = m.Var(name='alphaPROe_2NaveVarY7')
alphaPROe_1NaveVarY7  = m.Var(name='alphaPROe_1NaveVarY7')
alphaPROe4NaveVarY7   = m.Var(name='alphaPROe4NaveVarY7')
alphaPROe5NaveVarY7   = m.Var(name='alphaPROe5NaveVarY7')
alphaPROe6NaveVarY7   = m.Var(name='alphaPROe6NaveVarY7')
alphaPROe7NaveVarY7   = m.Var(name='alphaPROe7NaveVarY7')
elasticE_2NaveVarY7   = m.Var(name='elasticE_2NaveVarY7')
elasticE_1NaveVarY7   = m.Var(name='elasticE_1NaveVarY7')
elasticE4NaveVarY7    = m.Var(name='elasticE4NaveVarY7')
elasticE5NaveVarY7    = m.Var(name='elasticE5NaveVarY7')
elasticE6NaveVarY7    = m.Var(name='elasticE6NaveVarY7')
elasticE7NaveVarY7    = m.Var(name='elasticE7NaveVarY7')
PeakElasticNaveVarY7  = m.Var(name='PeakElasticNaveVarY7')
PeakAlphaNaveVarY7    = m.Var(name='PeakAlphaNaveVarY7')
PeakGANaveVarY7       = m.Var(name='PeakGANaveVarY7')
PeakGENaveVarY7       = m.Var(name='PeakGENaveVarY7')
costVarY7             = m.Var(name='PriceVarY7')

var7List = [densityVar7,densityVarY7,densityNaveVarY7,gamABe6VarY7,gamABe7NaveVarY7,gamPROe_2NaveVarY7,gamPROe_1NaveVarY7,\
            gamPROe4NaveVarY7,gamPROe5NaveVarY7,gamPROe7NaveVarY7,alphaPROe_2NaveVarY7,alphaPROe_1NaveVarY7,\
            alphaPROe4NaveVarY7,alphaPROe5NaveVarY7,alphaPROe6NaveVarY7,alphaPROe7NaveVarY7,elasticE_2NaveVarY7,\
            elasticE_1NaveVarY7,elasticE4NaveVarY7,elasticE5NaveVarY7,elasticE6NaveVarY7,elasticE7NaveVarY7,\
            PeakElasticNaveVarY7,PeakAlphaNaveVarY7,PeakGANaveVarY7,PeakGENaveVarY7,costVarY7]

# Layer 7 Splines
m.cspline(densityVar7, densityVarY7, IDs, densities)
m.cspline(densityVar7, gamABe6VarY7, IDs, LogXSgamABe6)
m.cspline(densityVar7, gamABe7NaveVarY7, IDs, LogXSgamABe7)
m.cspline(densityVar7, gamPROe_2NaveVarY7, IDs, LogXSgamPROe_2)
m.cspline(densityVar7, gamPROe_1NaveVarY7, IDs, LogXSgamPROe_1)
m.cspline(densityVar7, gamPROe4NaveVarY7, IDs, LogXSgamPROe4)
m.cspline(densityVar7, gamPROe5NaveVarY7, IDs, LogXSgamPROe5)
### This one was skipped because it's not in the model ###
m.cspline(densityVar7, gamPROe7NaveVarY7, IDs, LogXSgamPROe7)
m.cspline(densityVar7, alphaPROe_2NaveVarY7, IDs, LogXSalphaPROe_2)
m.cspline(densityVar7, alphaPROe_1NaveVarY7, IDs, LogXSalphaPROe_1)
m.cspline(densityVar7, alphaPROe4NaveVarY7, IDs, LogXSalphaPROe4)
m.cspline(densityVar7, alphaPROe5NaveVarY7, IDs, LogXSalphaPROe5)
m.cspline(densityVar7, alphaPROe6NaveVarY7, IDs, LogXSalphaPROe6)
m.cspline(densityVar7, alphaPROe7NaveVarY7, IDs, LogXSalphaPROe7)
m.cspline(densityVar7, elasticE_2NaveVarY7, IDs, LogXSelasticE_2)
m.cspline(densityVar7, elasticE_1NaveVarY7, IDs, LogXSelasticE_1)
m.cspline(densityVar7, elasticE4NaveVarY7, IDs, LogXSelasticE4)
m.cspline(densityVar7, elasticE5NaveVarY7, IDs, LogXSelasticE5)
m.cspline(densityVar7, elasticE6NaveVarY7, IDs, LogXSelasticE6)
m.cspline(densityVar7, elasticE7NaveVarY7, IDs, LogXSelasticE7)
m.cspline(densityVar7, PeakElasticNaveVarY7, IDs, LogPeakElastic)
m.cspline(densityVar7, PeakAlphaNaveVarY7, IDs, LogPeakAlpha)
m.cspline(densityVar7, PeakGANaveVarY7, IDs, LogPeakGA)
m.cspline(densityVar7, PeakGENaveVarY7, IDs, LogPeakGE)
m.cspline(densityVar7, costVarY7, IDs, costs)

### Layer 8 ###
# LAYER 8 VARIABLES
densityVar8           = m.Var(name='densityVar8',value=L8guess, lb=lwbd, ub=5, integer=True)
densityVarY8          = m.Var(name='densityVarY8')
densityNaveVarY8      = m.Intermediate(densityVarY8)
gamABe6VarY8          = m.Var(name='gamABe6VarY8')
gamABe7NaveVarY8      = m.Var(name='gamABe7NaveVarY8')
gamPROe_2NaveVarY8    = m.Var(name='gamPROe_2NaveVarY8')
gamPROe_1NaveVarY8    = m.Var(name='gamPROe_1NaveVarY8')
gamPROe4NaveVarY8     = m.Var(name='gamPROe4NaveVarY8')
gamPROe5NaveVarY8     = m.Var(name='gamPROe5NaveVarY8')
gamPROe7NaveVarY8     = m.Var(name='gamPROe7NaveVarY8')
alphaPROe_2NaveVarY8  = m.Var(name='alphaPROe_2NaveVarY8')
alphaPROe_1NaveVarY8  = m.Var(name='alphaPROe_1NaveVarY8')
alphaPROe4NaveVarY8   = m.Var(name='alphaPROe4NaveVarY8')
alphaPROe5NaveVarY8   = m.Var(name='alphaPROe5NaveVarY8')
alphaPROe6NaveVarY8   = m.Var(name='alphaPROe6NaveVarY8')
alphaPROe7NaveVarY8   = m.Var(name='alphaPROe7NaveVarY8')
elasticE_2NaveVarY8   = m.Var(name='elasticE_2NaveVarY8')
elasticE_1NaveVarY8   = m.Var(name='elasticE_1NaveVarY8')
elasticE4NaveVarY8    = m.Var(name='elasticE4NaveVarY8')
elasticE5NaveVarY8    = m.Var(name='elasticE5NaveVarY8')
elasticE6NaveVarY8    = m.Var(name='elasticE6NaveVarY8')
elasticE7NaveVarY8    = m.Var(name='elasticE7NaveVarY8')
PeakElasticNaveVarY8  = m.Var(name='PeakElasticNaveVarY8')
PeakAlphaNaveVarY8    = m.Var(name='PeakAlphaNaveVarY8')
PeakGANaveVarY8       = m.Var(name='PeakGANaveVarY8')
PeakGENaveVarY8       = m.Var(name='PeakGENaveVarY8')
costVarY8             = m.Var(name='PriceVarY8')

var8List = [densityVar8,densityVarY8,densityNaveVarY8,gamABe6VarY8,gamABe7NaveVarY8,gamPROe_2NaveVarY8,gamPROe_1NaveVarY8,\
            gamPROe4NaveVarY8,gamPROe5NaveVarY8,gamPROe7NaveVarY8,alphaPROe_2NaveVarY8,alphaPROe_1NaveVarY8,\
            alphaPROe4NaveVarY8,alphaPROe5NaveVarY8,alphaPROe6NaveVarY8,alphaPROe7NaveVarY8,elasticE_2NaveVarY8,\
            elasticE_1NaveVarY8,elasticE4NaveVarY8,elasticE5NaveVarY8,elasticE6NaveVarY8,elasticE7NaveVarY8,\
            PeakElasticNaveVarY8,PeakAlphaNaveVarY8,PeakGANaveVarY8,PeakGENaveVarY8,costVarY8]

# Layer 8 Splines
m.cspline(densityVar8, densityVarY8, IDs, densities)
m.cspline(densityVar8, gamABe6VarY8, IDs, LogXSgamABe6)
m.cspline(densityVar8, gamABe7NaveVarY8, IDs, LogXSgamABe7)
m.cspline(densityVar8, gamPROe_2NaveVarY8, IDs, LogXSgamPROe_2)
m.cspline(densityVar8, gamPROe_1NaveVarY8, IDs, LogXSgamPROe_1)
m.cspline(densityVar8, gamPROe4NaveVarY8, IDs, LogXSgamPROe4)
m.cspline(densityVar8, gamPROe5NaveVarY8, IDs, LogXSgamPROe5)
### This one was skipped because it's not in the model ###
m.cspline(densityVar8, gamPROe7NaveVarY8, IDs, LogXSgamPROe7)
m.cspline(densityVar8, alphaPROe_2NaveVarY8, IDs, LogXSalphaPROe_2)
m.cspline(densityVar8, alphaPROe_1NaveVarY8, IDs, LogXSalphaPROe_1)
m.cspline(densityVar8, alphaPROe4NaveVarY8, IDs, LogXSalphaPROe4)
m.cspline(densityVar8, alphaPROe5NaveVarY8, IDs, LogXSalphaPROe5)
m.cspline(densityVar8, alphaPROe6NaveVarY8, IDs, LogXSalphaPROe6)
m.cspline(densityVar8, alphaPROe7NaveVarY8, IDs, LogXSalphaPROe7)
m.cspline(densityVar8, elasticE_2NaveVarY8, IDs, LogXSelasticE_2)
m.cspline(densityVar8, elasticE_1NaveVarY8, IDs, LogXSelasticE_1)
m.cspline(densityVar8, elasticE4NaveVarY8, IDs, LogXSelasticE4)
m.cspline(densityVar8, elasticE5NaveVarY8, IDs, LogXSelasticE5)
m.cspline(densityVar8, elasticE6NaveVarY8, IDs, LogXSelasticE6)
m.cspline(densityVar8, elasticE7NaveVarY8, IDs, LogXSelasticE7)
m.cspline(densityVar8, PeakElasticNaveVarY8, IDs, LogPeakElastic)
m.cspline(densityVar8, PeakAlphaNaveVarY8, IDs, LogPeakAlpha)
m.cspline(densityVar8, PeakGANaveVarY8, IDs, LogPeakGA)
m.cspline(densityVar8, PeakGENaveVarY8, IDs, LogPeakGE)
m.cspline(densityVar8, costVarY8, IDs, costs)

### Layer 9 ###
# LAYER 9 VARIABLES
densityVar9           = m.Var(name='densityVar9',value=L9guess, lb=lwbd, ub=5, integer=True)
densityVarY9          = m.Var(name='densityVarY9')
densityNaveVarY9      = m.Intermediate(densityVarY9)
gamABe6VarY9          = m.Var(name='gamABe6VarY9')
gamABe7NaveVarY9      = m.Var(name='gamABe7NaveVarY9')
gamPROe_2NaveVarY9    = m.Var(name='gamPROe_2NaveVarY9')
gamPROe_1NaveVarY9    = m.Var(name='gamPROe_1NaveVarY9')
gamPROe4NaveVarY9     = m.Var(name='gamPROe4NaveVarY9')
gamPROe5NaveVarY9     = m.Var(name='gamPROe5NaveVarY9')
gamPROe7NaveVarY9     = m.Var(name='gamPROe7NaveVarY9')
alphaPROe_2NaveVarY9  = m.Var(name='alphaPROe_2NaveVarY9')
alphaPROe_1NaveVarY9  = m.Var(name='alphaPROe_1NaveVarY9')
alphaPROe4NaveVarY9   = m.Var(name='alphaPROe4NaveVarY9')
alphaPROe5NaveVarY9   = m.Var(name='alphaPROe5NaveVarY9')
alphaPROe6NaveVarY9   = m.Var(name='alphaPROe6NaveVarY9')
alphaPROe7NaveVarY9   = m.Var(name='alphaPROe7NaveVarY9')
elasticE_2NaveVarY9   = m.Var(name='elasticE_2NaveVarY9')
elasticE_1NaveVarY9   = m.Var(name='elasticE_1NaveVarY9')
elasticE4NaveVarY9    = m.Var(name='elasticE4NaveVarY9')
elasticE5NaveVarY9    = m.Var(name='elasticE5NaveVarY9')
elasticE6NaveVarY9    = m.Var(name='elasticE6NaveVarY9')
elasticE7NaveVarY9    = m.Var(name='elasticE7NaveVarY9')
PeakElasticNaveVarY9  = m.Var(name='PeakElasticNaveVarY9')
PeakAlphaNaveVarY9    = m.Var(name='PeakAlphaNaveVarY9')
PeakGANaveVarY9       = m.Var(name='PeakGANaveVarY9')
PeakGENaveVarY9       = m.Var(name='PeakGENaveVarY9')
costVarY9             = m.Var(name='PriceVarY9')

var9List = [densityVar9,densityVarY9,densityNaveVarY9,gamABe6VarY9,gamABe7NaveVarY9,gamPROe_2NaveVarY9,gamPROe_1NaveVarY9,\
            gamPROe4NaveVarY9,gamPROe5NaveVarY9,gamPROe7NaveVarY9,alphaPROe_2NaveVarY9,alphaPROe_1NaveVarY9,\
            alphaPROe4NaveVarY9,alphaPROe5NaveVarY9,alphaPROe6NaveVarY9,alphaPROe7NaveVarY9,elasticE_2NaveVarY9,\
            elasticE_1NaveVarY9,elasticE4NaveVarY9,elasticE5NaveVarY9,elasticE6NaveVarY9,elasticE7NaveVarY9,\
            PeakElasticNaveVarY9,PeakAlphaNaveVarY9,PeakGANaveVarY9,PeakGENaveVarY9,costVarY9]

# Layer 9 Splines
m.cspline(densityVar9, densityVarY9, IDs, densities)
m.cspline(densityVar9, gamABe6VarY9, IDs, LogXSgamABe6)
m.cspline(densityVar9, gamABe7NaveVarY9, IDs, LogXSgamABe7)
m.cspline(densityVar9, gamPROe_2NaveVarY9, IDs, LogXSgamPROe_2)
m.cspline(densityVar9, gamPROe_1NaveVarY9, IDs, LogXSgamPROe_1)
m.cspline(densityVar9, gamPROe4NaveVarY9, IDs, LogXSgamPROe4)
m.cspline(densityVar9, gamPROe5NaveVarY9, IDs, LogXSgamPROe5)
### This one was skipped because it's not in the model ###
m.cspline(densityVar9, gamPROe7NaveVarY9, IDs, LogXSgamPROe7)
m.cspline(densityVar9, alphaPROe_2NaveVarY9, IDs, LogXSalphaPROe_2)
m.cspline(densityVar9, alphaPROe_1NaveVarY9, IDs, LogXSalphaPROe_1)
m.cspline(densityVar9, alphaPROe4NaveVarY9, IDs, LogXSalphaPROe4)
m.cspline(densityVar9, alphaPROe5NaveVarY9, IDs, LogXSalphaPROe5)
m.cspline(densityVar9, alphaPROe6NaveVarY9, IDs, LogXSalphaPROe6)
m.cspline(densityVar9, alphaPROe7NaveVarY9, IDs, LogXSalphaPROe7)
m.cspline(densityVar9, elasticE_2NaveVarY9, IDs, LogXSelasticE_2)
m.cspline(densityVar9, elasticE_1NaveVarY9, IDs, LogXSelasticE_1)
m.cspline(densityVar9, elasticE4NaveVarY9, IDs, LogXSelasticE4)
m.cspline(densityVar9, elasticE5NaveVarY9, IDs, LogXSelasticE5)
m.cspline(densityVar9, elasticE6NaveVarY9, IDs, LogXSelasticE6)
m.cspline(densityVar9, elasticE7NaveVarY9, IDs, LogXSelasticE7)
m.cspline(densityVar9, PeakElasticNaveVarY9, IDs, LogPeakElastic)
m.cspline(densityVar9, PeakAlphaNaveVarY9, IDs, LogPeakAlpha)
m.cspline(densityVar9, PeakGANaveVarY9, IDs, LogPeakGA)
m.cspline(densityVar9, PeakGENaveVarY9, IDs, LogPeakGE)
m.cspline(densityVar9, costVarY9, IDs, costs)

### Recently Added ###
variableMats = []
variableMats.append(densityVar0)
variableMats.append(densityVar1)
variableMats.append(densityVar2)
variableMats.append(densityVar3)
variableMats.append(densityVar4)
variableMats.append(densityVar5)
variableMats.append(densityVar6)
variableMats.append(densityVar7)
variableMats.append(densityVar8)
variableMats.append(densityVar9)

### Define Test Array Function ###

def FillTestArray(magicNum,materialArray,layerPosition):
    
####### Unpack materialArray ######
    densityVar,densityVarY,densityNaveVarY,gamABe6VarY,gamABe7NaveVarY,gamPROe_2NaveVarY,gamPROe_1NaveVarY,\
            gamPROe4NaveVarY,gamPROe5NaveVarY,gamPROe7NaveVarY,alphaPROe_2NaveVarY,alphaPROe_1NaveVarY,\
            alphaPROe4NaveVarY,alphaPROe5NaveVarY,alphaPROe6NaveVarY,alphaPROe7NaveVarY,elasticE_2NaveVarY,\
            elasticE_1NaveVarY,elasticE4NaveVarY,elasticE5NaveVarY,elasticE6NaveVarY,elasticE7NaveVarY,\
            PeakElasticNaveVarY,PeakAlphaNaveVarY,PeakGANaveVarY,PeakGENaveVarY,costVarY = materialArray
    
####### Empty Array Definitions ######
    
    testArray = [[-999999] * 61]

#### Fill testArray i,0; use MatID to assign object ###
    
    testArray[0][0]  = densityVar

####### Calculate Some Constants ######
# Some required intermediates for true radius:
    Δradius90th = 0.018626667 
    innerRadius = 1.075 - Δradius90th
    outerRadius = 2.732773363
    Δradius = (outerRadius - innerRadius) / (magicNum)        

####### Fill first 8 features ######
    testArray[0][1]  = densityVarY                                     # Density
    testArray[0][2]  = -999                                            # PLACEHOLDER for Density Nave
    testArray[0][3]  = -999                                            # PLACEHOLDER for True Radius (m)
    testArray[0][4]  = -999                                            # PLACEHOLDER for Radius (m)  
    testArray[0][5]  = layerPosition                                   # cell count
    testArray[0][3]  = innerRadius + (layerPosition + 1) * Δradius     # True Radius (m) UPDATED
    testArray[0][4]  = Δradius*(layerPosition + 1)                     # Radius (m) UPDATED
    testArray[0][6]  = np.pi*( testArray[0][3]**2 - (innerRadius + Δradius*testArray[0][5] )**2 )  # Cross-Sectional Area (m^2)
    testArray[0][7]  = testArray[0][6]*2                                      # Volume (m^3)
    massVarY         = m.Intermediate(densityVarY*maxesForVarsOnly[0]*testArray[0][7]/maxVals[3])  # Equation for defining mass
    testArray[0][8]  = massVarY                                        # mass line of testArray

####### Assign LogXS features ######
    testArray[0][3]  = innerRadius + (layerPosition + 1) * Δradius / maxVals[1] # update true radius feature
    testArray[0][4]  = Δradius*(layerPosition + 1)  / maxVals[2]                # update radius feature
    testArray[0][9]  = gamABe6VarY                        # LogXSgamABe6
    testArray[0][10] = gamABe7NaveVarY                    # LogXSgamABe7
    testArray[0][11] = gamPROe_2NaveVarY                  # LogXSgamPROe-2
    testArray[0][12] = gamPROe_1NaveVarY                  # LogXSgamPROe-1
    testArray[0][13] = gamPROe4NaveVarY                   # LogXSgamPROe4
    testArray[0][14] = gamPROe5NaveVarY                   # LogXSgamPROe5
    testArray[0][15] = -999                               # LogXSgamPROe6
    testArray[0][16] = gamPROe7NaveVarY                   # LogXSgamPROe7
    testArray[0][17] = alphaPROe_2NaveVarY                # LogXSalphaPROe-2
    testArray[0][18] = alphaPROe_1NaveVarY                # LogXSalphaPROe-1
    testArray[0][19] = alphaPROe4NaveVarY                 # LogXSalphaPROe4
    testArray[0][20] = alphaPROe5NaveVarY                 # LogXSalphaPROe5
    testArray[0][21] = alphaPROe6NaveVarY                 # LogXSalphaPROe6
    testArray[0][22] = alphaPROe7NaveVarY                 # LogXSalphaPROe7
    testArray[0][23] = elasticE_2NaveVarY                 # LogXSelasticE-2
    testArray[0][24] = elasticE_1NaveVarY                 # LogXSelasticE-1
    testArray[0][25] = elasticE4NaveVarY                  # LogXSelasticE4
    testArray[0][26] = elasticE5NaveVarY                  # LogXSelasticE5
    testArray[0][27] = elasticE6NaveVarY                  # LogXSelasticE6
    testArray[0][28] = elasticE7NaveVarY                  # LogXSelasticE7
    testArray[0][29] = PeakElasticNaveVarY                # LogPeakElastic
    testArray[0][30] = PeakAlphaNaveVarY                  # LogPeakAlpha
    testArray[0][31] = PeakGANaveVarY                     # LogPeakGA
    testArray[0][32] = PeakGENaveVarY                     # LogPeakGE

### Calculate and fill Nave features ###    
    testArray[0][2]  = testArray[0][1]        # Density Nave UPDATED
    testArray[0][33] = testArray[0][9]        # LogXSgamABe6Nave
    testArray[0][34] = testArray[0][10]       # LogXSgamABe7Nave
    testArray[0][35] = testArray[0][11]       # LogXSgamPROe-2Nave
    testArray[0][36] = testArray[0][12]       # LogXSgamPROe-1Nave
    testArray[0][37] = testArray[0][13]       # LogXSgamPROe4Nave
    testArray[0][38] = testArray[0][14]       # LogXSgamPROe5Nave
    testArray[0][39] = testArray[0][15]       # LogXSgamPROe6Nave
    testArray[0][40] = testArray[0][16]       # LogXSgamPROe7Nave
    testArray[0][41] = testArray[0][17]       # LogXSalphaPROe-2Nave
    testArray[0][42] = testArray[0][18]       # LogXSalphaPROe-1Nave
    testArray[0][43] = testArray[0][19]       # LogXSalphaPROe4Nave
    testArray[0][44] = testArray[0][20]       # LogXSalphaPROe5Nave
    testArray[0][45] = testArray[0][21]       # LogXSalphaPROe6Nave
    testArray[0][46] = testArray[0][22]       # LogXSalphaPROe7Nave
    testArray[0][47] = testArray[0][23]       # LogXSelasticE-2Nave
    testArray[0][48] = testArray[0][24]       # LogXSelasticE-1Nave
    testArray[0][49] = testArray[0][25]       # LogXSelasticE4Nave
    testArray[0][50] = testArray[0][26]       # LogXSelasticE5Nave
    testArray[0][51] = testArray[0][27]       # LogXSelasticE6Nave
    testArray[0][52] = testArray[0][28]       # LogXSelasticE7Nave
    testArray[0][53] = testArray[0][29]       # LogPeakElasticNave
    testArray[0][54] = testArray[0][30]       # LogPeakAlphaNave
    testArray[0][55] = testArray[0][31]       # LogPeakGANave
    testArray[0][56] = testArray[0][32]       # LogPeakGENave
    
### Assign Line Cost ###    
    testArray[0][57] = costVarY               # line cost
    
    return testArray

#### Call Function, get input_data ###

input_data0 = FillTestArray(magicNum,var0List,0)
input_data1 = FillTestArray(magicNum,var1List,1)
input_data2 = FillTestArray(magicNum,var2List,2)
input_data3 = FillTestArray(magicNum,var3List,3)
input_data4 = FillTestArray(magicNum,var4List,4)
input_data5 = FillTestArray(magicNum,var5List,5)
input_data6 = FillTestArray(magicNum,var6List,6)
input_data7 = FillTestArray(magicNum,var7List,7)
input_data8 = FillTestArray(magicNum,var8List,8)
input_data9 = FillTestArray(magicNum,var9List,9)


### Predict Dosage #######

predictedDose = []

# Indices to drop, so that the array is features only
indices_to_drop = [0, 2, 5, 6, 7] + list(range(9, 33)) + [39] + list(range(57, 61))

# Filtering data
filtered_data0 = [item for index, item in enumerate(input_data0[0]) if index not in indices_to_drop]
filtered_data1 = [item for index, item in enumerate(input_data1[0]) if index not in indices_to_drop]
filtered_data2 = [item for index, item in enumerate(input_data2[0]) if index not in indices_to_drop]
filtered_data3 = [item for index, item in enumerate(input_data3[0]) if index not in indices_to_drop]
filtered_data4 = [item for index, item in enumerate(input_data4[0]) if index not in indices_to_drop]
filtered_data5 = [item for index, item in enumerate(input_data5[0]) if index not in indices_to_drop]
filtered_data6 = [item for index, item in enumerate(input_data6[0]) if index not in indices_to_drop]
filtered_data7 = [item for index, item in enumerate(input_data7[0]) if index not in indices_to_drop]
filtered_data8 = [item for index, item in enumerate(input_data8[0]) if index not in indices_to_drop]
filtered_data9 = [item for index, item in enumerate(input_data9[0]) if index not in indices_to_drop]

# Use loaded model to predict the dose after this layer
predictedDose.append(Model(filtered_data0)[0])
predictedDose.append(Model(filtered_data1)[0])
predictedDose.append(Model(filtered_data2)[0])
predictedDose.append(Model(filtered_data3)[0])
predictedDose.append(Model(filtered_data4)[0])
predictedDose.append(Model(filtered_data5)[0])
predictedDose.append(Model(filtered_data6)[0])
predictedDose.append(Model(filtered_data7)[0])
predictedDose.append(Model(filtered_data8)[0])
predictedDose.append(Model(filtered_data9)[0])

### Objective function and Constraints #####
# Print Initial Values
print('Initial Material Variables:',variableMats)

# Objective value
objectiveToMinimize = predictedDose[-1]

# This is to minimize the dose after the last layer of shield
m.Minimize(objectiveToMinimize)

# Added constraints
mass0        = input_data0[0][8]*maxVals[3]
mass1        = input_data1[0][8]*maxVals[3]
mass2        = input_data2[0][8]*maxVals[3]
mass3        = input_data3[0][8]*maxVals[3]
mass4        = input_data4[0][8]*maxVals[3]
mass5        = input_data5[0][8]*maxVals[3]
mass6        = input_data6[0][8]*maxVals[3]
mass7        = input_data7[0][8]*maxVals[3]
mass8        = input_data8[0][8]*maxVals[3]
mass9        = input_data9[0][8]*maxVals[3]

cost0        = m.Intermediate(mass0*input_data0[0][57])
cost1        = m.Intermediate(mass1*input_data1[0][57])
cost2        = m.Intermediate(mass2*input_data2[0][57])
cost3        = m.Intermediate(mass3*input_data3[0][57])
cost4        = m.Intermediate(mass4*input_data4[0][57])
cost5        = m.Intermediate(mass5*input_data5[0][57])
cost6        = m.Intermediate(mass6*input_data6[0][57])
cost7        = m.Intermediate(mass7*input_data7[0][57])
cost8        = m.Intermediate(mass8*input_data8[0][57])
cost9        = m.Intermediate(mass9*input_data9[0][57])

massTotalSum = m.Intermediate(mass0 + mass1 + mass2 + mass3 + mass4 + mass5 + mass6 + mass7 + mass8 + mass9)
                             
costTotalSum = m.Intermediate(cost0 + cost1 + cost2 + cost3 + cost4 + cost5 + cost6 + cost7 + cost8 + cost9)

doseLimit = -0.90

m.Equation(massTotalSum <= massLimit)
m.Equation(costTotalSum <= costLimit)
m.Equation(objectiveToMinimize <= doseLimit)

#### Solve Step ###  
m.options.REDUCE = 3
m.options.SOLVER = 1
m.solver_options = ['minlp_gap_tol 1.0e-2',\
                    'minlp_maximum_iterations 10000',\
                    'minlp_max_iter_with_int_sol 500',\
                    'minlp_branch_method 1',\
                    'nlp_maximum_iterations 20']
m.solve(disp=True)

#### Extract optimized values ###  
print('Predicted Dose:',predictedDose)

# Layer 0:
variablesList = [ densityVar0.value[0],densityVarY0.value[0],densityNaveVarY0.value[0], input_data0[0][3],input_data0[0][4],\
      input_data0[0][8].value[0]*maxVals[4],gamABe6VarY0.value[0],gamABe7NaveVarY0.value[0],gamPROe_2NaveVarY0.value[0],\
      gamPROe_1NaveVarY0.value[0],gamPROe4NaveVarY0.value[0],\
      gamPROe5NaveVarY0.value[0],gamPROe7NaveVarY0.value[0],alphaPROe_2NaveVarY0.value[0],alphaPROe_1NaveVarY0.value[0],\
      alphaPROe4NaveVarY0.value[0],alphaPROe5NaveVarY0.value[0],\
      alphaPROe6NaveVarY0.value[0],alphaPROe7NaveVarY0.value[0],elasticE_2NaveVarY0.value[0],elasticE_1NaveVarY0.value[0],\
      elasticE4NaveVarY0.value[0],elasticE5NaveVarY0.value[0],\
      elasticE6NaveVarY0.value[0],elasticE7NaveVarY0.value[0],PeakElasticNaveVarY0.value[0],PeakAlphaNaveVarY0.value[0],\
      PeakGANaveVarY0.value[0],PeakGENaveVarY0.value[0],costVarY0.value[0] ]

printListfeatures = ['Material Code','Density','Density Nave','True Radius (m)','Radius (m)','Mass (kg)','LogXSgamABe6Nave',\
            'LogXSgamABe7Nave','LogXSgamPROe-2Nave',\
            'LogXSgamPROe-1Nave','LogXSgamPROe4Nave','LogXSgamPROe5Nave','LogXSgamPROe7Nave',\
            'LogXSalphaPROe-2Nave','LogXSalphaPROe-1Nave',\
            'LogXSalphaPROe4Nave','LogXSalphaPROe5Nave','LogXSalphaPROe6Nave','LogXSalphaPROe7Nave','LogXSelasticE-2Nave',\
            'LogXSelasticE-1Nave','LogXSelasticE4Nave',\
            'LogXSelasticE5Nave','LogXSelasticE6Nave','LogXSelasticE7Nave','LogPeakElasticNave','LogPeakAlphaNave',\
            'LogPeakGANave','LogPeakGENave']

print('Variables Final Values:\n')
for i in range(len(features)):
    print(printListfeatures[i],':',variablesList[i])

# Layer 1
variablesList1 = [ densityVar1.value[0],densityVarY1.value[0],densityNaveVarY1.value[0], input_data1[0][3],input_data1[0][4],\
      input_data1[0][8].value[0]*maxVals[4],gamABe6VarY1.value[0],gamABe7NaveVarY1.value[0],gamPROe_2NaveVarY1.value[0],\
      gamPROe_1NaveVarY1.value[0],gamPROe4NaveVarY1.value[0],\
      gamPROe5NaveVarY1.value[0],gamPROe7NaveVarY1.value[0],alphaPROe_2NaveVarY1.value[0],alphaPROe_1NaveVarY1.value[0],\
      alphaPROe4NaveVarY1.value[0],alphaPROe5NaveVarY1.value[0],\
      alphaPROe6NaveVarY1.value[0],alphaPROe7NaveVarY1.value[0],elasticE_2NaveVarY1.value[0],elasticE_1NaveVarY1.value[0],\
      elasticE4NaveVarY1.value[0],elasticE5NaveVarY1.value[0],\
      elasticE6NaveVarY1.value[0],elasticE7NaveVarY1.value[0],PeakElasticNaveVarY1.value[0],PeakAlphaNaveVarY1.value[0],\
      PeakGANaveVarY1.value[0],PeakGENaveVarY1.value[0],costVarY1.value[0] ]

printListfeatures = ['Material Code','Density','Density Nave','True Radius (m)','Radius (m)','Mass (kg)','LogXSgamABe6Nave',\
            'LogXSgamABe7Nave','LogXSgamPROe-2Nave',\
            'LogXSgamPROe-1Nave','LogXSgamPROe4Nave','LogXSgamPROe5Nave','LogXSgamPROe7Nave',\
            'LogXSalphaPROe-2Nave','LogXSalphaPROe-1Nave',\
            'LogXSalphaPROe4Nave','LogXSalphaPROe5Nave','LogXSalphaPROe6Nave','LogXSalphaPROe7Nave','LogXSelasticE-2Nave',\
            'LogXSelasticE-1Nave','LogXSelasticE4Nave',\
            'LogXSelasticE5Nave','LogXSelasticE6Nave','LogXSelasticE7Nave','LogPeakElasticNave','LogPeakAlphaNave',\
            'LogPeakGANave','LogPeakGENave']

print('Variables Final Values:\n')
for i in range(len(features)):
    print(printListfeatures[i],':',variablesList1[i])

# Overall Stats:
print()
print('Total Volume:',input_data0[0][7]+input_data1[0][7],'m^3')
print('Total Mass:',round(massTotalSum.value[0],0),'kg')
print('Total Cost: $',round(costTotalSum.value[0],2))
print('Final Material Variables:',variableMats)
