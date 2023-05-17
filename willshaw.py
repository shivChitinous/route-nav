import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

def evolve_willshaw(I, w1, w2, KCthresh):
    
    #generate KC firing
    KC = (np.matmul(w1.T,I)>KCthresh).astype('int')
    
    MBON = np.matmul(w2.T,KC)

    return KC, MBON

def init_willshaw(nVPNs, nKCs, perKC, seed=0):
    
    #get weight 1
    w1 = np.zeros((nVPNs, nKCs))
    np.random.seed(seed)
    for kc in range(nKCs):
        w1[np.random.choice(nVPNs, perKC),kc] = 1
        
    #get weight 2
    w2 = np.ones((nKCs,1))
    
    return w1, w2

def learn_willshaw(KC, w2):
    #set
    idx = KC.astype('bool')
    w2[idx] = 0
    return w2

def learn_w(X):
    return 1-np.max(X,axis=1) #willshaw learning rule

def y(x,w):
    return np.matmul(w.T,x)

def novelty(X,x):
    return np.min(np.sum((X-x)**2,axis=0))

def output_novelty(X,Xrandom,w,plot=False,nbins=10,scatter=False,ax=None):
    n_random = Xrandom.shape[1]
    n = np.zeros(n_random)
    ny = np.zeros(n_random)
    for i in range(n_random):
        x = Xrandom[:,i].reshape(-1,1)
        n[i] = novelty(X,x)
        ny[i] = y(x,w)
    
    if plot:
        if scatter:
            ax.plot(n,ny,'o',color='grey',alpha=0.5);

        N, _ = np.histogram(n, bins=nbins)
        sy, _ = np.histogram(n, bins=nbins, weights=ny)
        sy2, _ = np.histogram(n, bins=nbins, weights=ny*ny)
        mean = sy / N
        std = np.sqrt(sy2/N - mean*mean)

        ax.errorbar((_[1:] + _[:-1])/2, mean, yerr=std, fmt='o-')

    return n, ny, ax

def distort(XimS, k, seed=0):
    picsize = len(XimS)
    synapses = np.array(np.meshgrid(range(picsize),range(picsize))).T.reshape(-1,2)
    rng = np.random.default_rng(seed)
    modsyn = rng.choice(synapses, int(k*np.prod(np.shape(XimS))/100), 
                        axis=0, replace=False);
    XimDist = XimS.copy()
    XimDist[modsyn[:,0], modsyn[:,1]] = 1-XimS[modsyn[:,0], modsyn[:,1]]
    return XimDist

def distort_by_color(XimS, k, color=1, seed=0):
    picsize = len(XimS)
    synapses = np.array(np.meshgrid(range(picsize),range(picsize))).T.reshape(-1,2)
    rng = np.random.default_rng(seed)
    modsyn = rng.choice(synapses, int(k*np.prod(np.shape(XimS))/100), 
                        axis=0, replace=False);
    XimDist = XimS.copy()
    condition = XimS[modsyn[:,0], modsyn[:,1]]==color
    XimDist[modsyn[condition,0], modsyn[condition,1]] = 1-XimS[modsyn[condition,0], modsyn[condition,1]]
    return XimDist

def descendMBONgrad(grid, scan, scanI, w1, w2t, KCthresh):
    MBON = np.zeros([len(grid),len(scan)])
    descentDir = np.zeros(len(grid))
    
    i = 0
    for f in range(len(scan)):
        for l in range(len(grid)):
            _, MBON[l,f] = evolve_willshaw(scanI[:,i], w1, w2t, KCthresh)
            descentDir[l] = scan[np.argmin(MBON[l,:])]
            i += 1
    
    return MBON, descentDir