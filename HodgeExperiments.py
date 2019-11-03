import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from Hodge import *
    

def getConsistencyRatios(Y, I, H, W, verbose=False):
    normD0s = getWNorm(Y-H-I, W)
    
    [normY, normI, normH] = [getWNorm(Y, W), getWNorm(I, W), getWNorm(H, W)]
    a = (normD0s/normY)**2
    b = (normI/normY)**2
    c = (normH/normY)**2
    if verbose:
        print("|D0s/Y| = %g"%a)
        print("Local Inconsistency = %g"%b)
        print("Global Inconsistency = %g"%c)
        print("a + b + c = %g"%(a + b + c))
    return (a, b, c)

def doTotalOrderExperiment(N, NComparisons, NQuant = -1):
    """
    Do an experiment comparing binary weights on a total order
    to real weights
    Parameters
    ----------
    N: int
        Number of items to consider
    NComparisons: int
        Number of comparisons performed
    NQuant: int
        Number of quantization levels
    """
    I, J = np.meshgrid(np.arange(N), np.arange(N))
    I = I[np.triu_indices(N, 1)]
    J = J[np.triu_indices(N, 1)]
    #[I, J] = [I[0:N-1], J[0:N-1]]
    NEdges = len(I)
    R = np.zeros((NEdges, 2))
    R[:, 0] = J
    R[:, 1] = I
    
    #W = np.random.rand(NEdges)
    W = np.ones(NEdges)
    
    #Note: When using binary weights, Y is not necessarily a cocycle
    Y = I - J
    if NQuant > -1:
        Y = NQuant*np.round(NQuant*Y/N)
    idx = np.random.permutation(Y.size)[0:NComparisons]
    R = R[idx, :]
    W = W[idx]
    Y = Y[idx]
        
    (s, I, H) = doHodge(R, W, Y, verbose=False)
    return getConsistencyRatios(Y, I, H, W)

if __name__ == '__main__':
    np.random.seed(0)
    N = 50
    NTrials = 10
    AllNComparisons = np.arange(100, 1000, 10)
    Is = np.zeros((AllNComparisons.size, NTrials))
    Hs = np.zeros_like(Is)
    NLevels = 3
    for i, NComparisons in enumerate(AllNComparisons):
        for j in range(NTrials):
            d0, IMag, HMag = doTotalOrderExperiment(N, NComparisons, NLevels)
            Is[i, j] = IMag
            Hs[i, j] = HMag
        print("%i, local:%.3g, global:%.3g"%(NComparisons, np.mean(Is[i, :]), np.mean(Hs[i, :])))
    AllNComparisons = AllNComparisons[:, None]*np.ones((1, NTrials))
    sns.lineplot(AllNComparisons.flatten(), Is.flatten())
    sns.lineplot(AllNComparisons.flatten(), Hs.flatten())
    plt.xlabel("Number of Comparisons")
    plt.ylabel("Inconsistency Ratio")
    plt.legend(["Local Inconsistencies", "Global Inconsistencies"])
    plt.title("%i Level Inconsistencies for %i Objects"%(NLevels, N))
    plt.show()


if __name__ == '__main__2':
    np.random.seed(17)
    R = sio.loadmat('R.mat')['R']
    [R, Y] = [R[:, 0:2], R[:, 2]]
    W = np.random.rand(len(Y))
    #W = np.ones(len(Y))
    (s, I, H) = doHodge(R, W, Y)
    print(np.argsort(s))
    
    getConsistencyRatios(Y, I, H, W, verbose=True)
