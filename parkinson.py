import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

f = open("parkinsons.data","r")
header = f.readline()
names = header.strip().split(',')[1:]

data = np.loadtxt(f ,delimiter=',', usecols=1+np.arange(23))

targetColumn = names.index("status")
XColumns = np.arange(23)
XColumns = np.delete(XColumns, targetColumn)
X = data[:,XColumns]
T = data[:,targetColumn].reshape((-1,1)) # to keep 2-d matrix form
names.remove("status")
X.shape, T.shape

print('{:20s} {:^9s} {:^9s}'.format(' ','mean','stdev'))
for i in range(X.shape[1]):
    print('{:20s} {:9.3g} {:9.3g}'.format(names[i],np.mean(X[:,i]),np.std(X[:,i])))

uniq = np.unique(T)
print('   Value  Occurrences')
for i in uniq:
    print('{:7.1g} {:10d}'.format(i, np.sum(T==i)))
    
trainf = 0.8
healthyI,_ = np.where(T == 0)
parkI,_ = np.where(T == 1)
healthyI = np.random.permutation(healthyI)
parkI = np.random.permutation(parkI)

nHealthy = round(trainf*len(healthyI))
nPark = round(trainf*len(parkI))
rowsTrain = np.hstack((healthyI[:nHealthy], parkI[:nPark]))
Xtrain = X[rowsTrain,:]
Ttrain = T[rowsTrain,:]
rowsTest = np.hstack((healthyI[nHealthy:], parkI[nPark:]))
Xtest =  X[rowsTest,:]
Ttest =  T[rowsTest,:]

print
print('Xtrain is {:d} by {:d}. Ttrain is {:d} by {:d}'.format(*(Xtrain.shape + Ttrain.shape)))
uniq = np.unique(Ttrain)
print('   Value  Occurrences')
for i in uniq:
    print('{:7.1g} {:10d}'.format(i, np.sum(Ttrain==i)))

    
print('Xtest is {:d} by {:d}. Ttest is {:d} by {:d}'.format(*(Xtest.shape + Ttest.shape)))
uniq = np.unique(Ttest)
print('   Value  Occurrences')
for i in uniq:
    print('{:7.1g} {:10d}'.format(i, np.sum(Ttest==i)))

def makeStandardize(X):
    means = X.mean(axis=0)
    stds = X.std(axis=0)
    def standardize(origX):
        return (origX - means) / stds
    def unStandardize(stdX):
        return stds * stdX + means
    return (standardize, unStandardize)

def normald(X, mu=None, sigma=None):
    """ normald:
       X contains samples, one per row, NxD. 
       mu is mean vector, Dx1.
       sigma is covariance matrix, DxD.  """
    d = X.shape[1]
    if np.any(mu == None):
        mu = np.zeros((d,1))
    if np.any(sigma == None):
        sigma = np.eye(d)
    detSigma = sigma if d == 1 else np.linalg.det(sigma)
    if detSigma == 0:
        raise np.linalg.LinAlgError('normald(): Singular covariance matrix')
    sigmaI = 1.0/sigma if d == 1 else np.linalg.inv(sigma)
    normConstant = 1.0 / np.sqrt((2*np.pi)**d * detSigma)
    diffv = X - mu.T # change column vector mu to be row vector
    return normConstant * np.exp(-0.5 * np.sum(np.dot(diffv, sigmaI) * diffv, axis=1))[:,np.newaxis]

#################################QDA########################################

Ttrain = Ttrain[:,-1]
Cu = list(np.unique(Ttrain))

# model = makeQDA(X,T)
def makeQDA(X,T):
    
    # Standardize X
    standardize,_ = makeStandardize(X)
    Xs = standardize(X)
    
    ClassesUnique = list(set(T))
    mu=[]
    sigma=[]
    prior=[]
    N = len(T)
    
    for classes in ClassesUnique:
        classrows1 = T==classes 
        mu1 = np.mean(Xs[classrows1,:],axis=0)             
        sigma1 = np.cov(Xs[classrows1,:].T)      
        NumEachClass = np.sum(classrows1)
        prior1 = NumEachClass / float(N)
        
        mu.append(mu1)
        sigma.append(sigma1)
        prior.append(prior1)
    
    mu = np.array(mu)
    sigma = np.array(sigma)        
    prior = np.array(prior)
    ClassesUnique = np.array(ClassesUnique)
    return {'mu':mu, 'sigma':sigma, 'prior':prior, 'UniqueClasses':ClassesUnique, 'standardize':standardize}

# QDA
def discQDA(X, standardize, mu, Sigma, prior):
    Xc = standardize(X) - mu
    if Sigma.size == 1:
        Sigma = np.asarray(Sigma).reshape((1,1))
    det = np.linalg.det(Sigma)        
    if det == 0:
        raise np.linalg.LinAlgError('discQDA(): Singular covariance matrix')
    SigmaInv = np.linalg.inv(Sigma)     # pinv in case Sigma is singular
    return -0.5 * np.log(det) \
           - 0.5 * np.sum(np.dot(Xc,SigmaInv) * Xc, axis=1) \
           + np.log(prior)
     
# predictedClass,classProbabilities,discriminantValues = useQDA(model,X)
def useQDA(model, X):    
    disc = []
    for i in range(len(model['UniqueClasses'])):
        mu = model['mu'][i]
        sigma = model['sigma'][i]
        prior = model['prior'][i]                        
        standardize = model['standardize']
        disc.append(discQDA(X, standardize, mu, sigma, prior))
    return(model['UniqueClasses'], model['prior'], np.array(disc))
    
model1 = makeQDA(Xtrain,Ttrain)
pC1,cP1,dV1 = useQDA(model1,Xtest)
pC2,cP2,dV2 = useQDA(model1,Xtrain)

pTestq = np.argmax(dV1,axis=0)
pTtrainq = np.argmax(dV2,axis=0)

def percentCorrect(p,t):
    return np.sum(p.ravel()==t.ravel()) / float(len(t)) * 100
    
print("QDA: Percent correct: Train {:.3g} Test {:.3g}".format(percentCorrect(pTtrainq,Ttrain),percentCorrect(pTestq,Ttest)))

######################################LDA################################################
# model = makeLDA(X,T)
def makeLDA(X,T):
    
    # Standardize X
    standardize,_ = makeStandardize(X)
    Xs = standardize(X)
    
    ClassesUnique = list(set(T))
    mu=[]
    sigma=0
    prior=[]
    N = len(T)
    
    for classes in ClassesUnique:
        classrows1 = T==classes 
        mu1 = np.mean(Xs[classrows1,:],axis=0)             
        sigma1 = np.cov(Xs[classrows1,:].T)      
        NumEachClass = np.sum(classrows1)
        prior1 = NumEachClass / float(N)
        
        mu.append(mu1)
        sigma = sigma + (prior1 * sigma1)
        prior.append(prior1)
    
    mu = np.array(mu)
    sigma = np.array(sigma)        
    prior = np.array(prior)
    ClassesUnique = np.array(ClassesUnique)
    return {'mu':mu, 'sigma':sigma, 'prior':prior, 'UniqueClasses':ClassesUnique, 'standardize':standardize}

# LDA
def discLDA(X, standardize, mu, Sigma, prior):
    Xc = standardize(X)
    if Sigma.size == 1:
        Sigma = np.asarray(Sigma).reshape((1,1))
    det = np.linalg.det(Sigma)        
    if det == 0:
        raise np.linalg.LinAlgError('discLDA(): Singular covariance matrix')
    SigmaInv = np.linalg.inv(Sigma)     # pinv in case Sigma is singular
    #print ((np.dot(SigmaInv, mu)))
    mu = mu.reshape((-1,1))
    return np.sum(Xc.T * np.dot(SigmaInv, mu), axis=0) \
           - (0.5 * np.dot(np.dot(mu.T, SigmaInv), mu)) \
           + np.log(prior)
     
# predictedClass,classProbabilities,discriminantValues = useQDA(model,X)
def useLDA(model, X):    
    disc = []
    for i in range(len(model['UniqueClasses'])):
        mu = model['mu'][i]
        sigma = model['sigma']
        prior = model['prior'][i]                        
        standardize = model['standardize']
        disc.append(discLDA(X, standardize, mu, sigma, prior))
    return(model['UniqueClasses'], model['prior'], np.array(disc))
modelda1=makeLDA(Xtrain, Ttrain)
pC3,cP3,dV3 = useQDA(model1,Xtest)
pC4,cP4,dV4 = useQDA(model1,Xtrain)

pTestl = np.argmax(dV3,axis=0)
pTtrainl = np.argmax(dV4,axis=0)
print("QDA: Percent correct: Train {:.3g} Test {:.3g}".format(percentCorrect(pTtrainl,Ttrain),percentCorrect(pTestl,Ttest)))

########################################Non-Linear Neural Network#####################################

import neuralnetworks1 as nn
import mpl_toolkits.mplot3d as plt3
from matplotlib import cm

Ttrain = Ttrain.reshape(156,1)
nHidden = 100
nnet = nn.NeuralNetworkClassifier(Xtrain.shape[1],nHidden,len(np.unique(Ttrain))) # 3 classes, will actually make 2-unit output layer
nnet.train(Xtrain, Ttrain,  nIterations=100)

pTnn,prnn,Zn = nnet.use(Xtrain,allOutputs=True)
pTnn2,prnn2,Zn2 = nnet.use(Xtest,allOutputs=True)

###################################Linear Logistic Regression###################################################

import n as nn2
def makeLLR(X,T):
    nHidden = X.shape[1]
    nnet1 = nn2.NeuralNetworkClassifier(X.shape[1],nHidden,len(np.unique(T))) 
    nnet1.train(X, T, nIterations=100)
    return nnet1

def useLLR(model,X):
    predTest4, probs4, Z4 = model.use(X,allOutputs=True)
    return predTest4, probs4, Z4
model4 = makeLLR(Xtrain,Ttrain)
pC,pP,Z = useLLR(model4,Xtrain)
pC2,pP2,Z2 = useLLR(model4, Xtest)
