import torch
import torch.nn.functional as F
from torch.distributions import RelaxedOneHotCategorical

import torch
import matplotlib.pyplot as plt
import numpy as np


def synthMRWregul_Torch(noise1,noise2,N,c1,c2,L,epsilon=0.2,win=1):
    
    '''
    sig=synthMRWregul(N,c1,c2,epsilon,L,win)
    
    Synthesis of a multifractal  motion with a singularity spevtrum
    zeta(q)=(c1+c2)q-c2 q^2/2. 
    
    Input :
        noise1 : stochastic noise of size N (tensor type)
        noise2 : stochastic noise of size N (tensor type)
        N : size of the signal to generate (tensor type)
        c1 : parameter of long-range dependance (tensor type)
        c2 : parameter of intermittency (tensor type)
        epsilon : size of the small-scale regularization (typically epsilon=2)
              Default value is 0.2) (tensor type)
         L : size of the integral scale. L must verify L<N/16. 
              Default value is N/16) (tensor type)
         win : type of the large scale regularization function (tensor type)
              if win==1 : gaussian function
              if win==2 : bump function
              Defaultvalue is 1.
              
    Output :
        sig : the synthetized fBm
        
    Example  :
        H=1/3;
        N=2^25;
        epsilon=2;
        L=2^14;
        sig=synthfbmRegul(N,H,L,epsilon)
        plot(sig)
        
    %%
    % S. Roux, ENS Lyon, 03/09/2017
    % Pythorch version C. Granero Belinchon, IMT Atlantique 01/2024
    '''
    
    # Regularised norm
    #RegulNorm=inline('sqrt(x.^2+epsilon^2)','x','epsilon'); % norme regularisé
    RegulNorm = lambda x, epsilon : torch.sqrt(x**2+epsilon**2)
    
    # Convert to tensor if needed
    if not isinstance(N, torch.Tensor):
        N = torch.tensor(N, dtype=torch.float32)
    if not isinstance(L, torch.Tensor):
        L = torch.tensor(L, dtype=torch.float32)
    if not isinstance(epsilon, torch.Tensor):
        epsilon = torch.tensor(epsilon, dtype=torch.float32)
    if not isinstance(c1, torch.Tensor):
        c1 = torch.tensor(c1, dtype=torch.float32)
    if not isinstance(c2, torch.Tensor):
        c2 = torch.tensor(c2, dtype=torch.float32)
    
    dx=1/N
    L=L/N
    epsilon=epsilon*dx
    alpha=3/2-c1
    
    N_int = int(N.item())
    x=torch.zeros(N_int,)
    n2=torch.zeros(N_int,)

    # Definition of x-axis
    x[0:int(N_int/2)+1]=torch.arange(0,N_int/2+1,1)*dx
    x[int(N_int/2)+1:N_int]=torch.arange(-N_int/2+1,0,1)*dx
    
    # Scaling
    y=x/RegulNorm(x,epsilon)**alpha
    
    # Large scale function
    if win==1: # Gaussian with std=L
        if L > 1/8:
            print('L must be greater than N/8 for right scaling')
        
        G=1/torch.sqrt(2*torch.tensor(torch.pi)*L**2)*torch.exp(-x**2/2/L**2)
    if win==2: # exp(x^/(L^2-x^2))
        if L > 1/2:
            print('L must be greater than N/2 for right scaling')
        
        G=x*0
        ii=torch.where(abs(x)<L)
        G[ii]=torch.exp(-(x[ii])**2/(L**2-x[ii]**2))
        G=G/sum(G)
    
    #------ Synthesis of the covariance of the correlated noise c2*log(|x|)
    if c2 >0:
        L_orig=L*N_int
        n2[0:int(N_int/2)+1]=torch.minimum(torch.arange(0,N_int/2+1,1).float(),L_orig)
        n2[int(N_int/2)+1:N_int]=torch.maximum(torch.arange(-N_int/2+1,0,1).float(),-L_orig)
    
        mycorr=c2*torch.log(L_orig/RegulNorm(abs(n2),1))
        L2=torch.real(torch.fft.fft(mycorr))
        x2=torch.real(torch.fft.ifft(torch.fft.fft(noise1)*torch.sqrt(L2)))
        
        Xr =torch.exp(x2-torch.var(x2)/2)
        
    else:
        Xr=torch.ones(N_int,)
    
    
    # -- synthesis of MRW
    bb=noise2
    sig =torch.real(torch.fft.ifft(torch.fft.fft(bb*torch.sqrt(dx)*Xr)*torch.fft.fft(y*G)))
    sig=sig/torch.std(sig)
    
    return sig

def analyseIncrsTorchcuda(signal,scales, device='cpu'):

    '''
    signal is the signal of study and scales is an array with the values of the scales of analysis
    '''  
    
    Nreal=signal.size()[0]
    Struc=torch.zeros((Nreal,3,len(scales)), dtype=torch.float32, device=device)
        
    for ir in range(Nreal):
        
        # We normalize the image by centering and standarizing it
        nanstdtmp=torch.sqrt(torch.nanmean(torch.abs(signal[ir]-torch.nanmean(signal[ir]))**2))
        tmp=(signal[ir]-torch.nanmean(signal[ir]))/nanstdtmp   

        for isc in range(len(scales)):
                
            scale=int(scales[isc])
                
            incrs=tmp[scale:]-tmp[:-scale]
            incrs=incrs[~torch.isnan(incrs)]
            Struc[ir,0,isc]=torch.log(torch.nanmean(incrs.flatten()**2))
            nanstdincrs=torch.sqrt(torch.nanmean(torch.abs(incrs-torch.nanmean(incrs))**2))
            incrsnorm=(incrs-torch.nanmean(incrs))/nanstdincrs
            Struc[ir,1,isc]=torch.nanmean(incrsnorm.flatten()**3)
            Struc[ir,2,isc]=torch.nanmean(incrsnorm.flatten()**4)/3
        
    return Struc


