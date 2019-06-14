
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This program implements the Primal Dual algorithm, presented by Chambolle-Pock with different
regularization function
Strasbourg: 20/03/2017
Cherni Afef
Modified : 07/06/2017
update   : 09/11/2017
"""
from __future__ import division, print_function
import sys
import numpy as np
import scipy
import matplotlib.pylab as plt
import matplotlib
import time

def prox_l0(x,gamma=1):
    """
    Computes proximity operator of l0 function
    """
    p = x * (np.abs(x)>= np.sqrt(2*gamma))
    return p 

def prox_logsum(x,gamma,w):
    """
    Compute proximity operator of log sum function
    """
    abx = np.abs(x)
    p = 0.5 * np.sign(x)*(abx-w+ np.sqrt((abx+w)**2-4*gamma))
    pos = 2*np.sqrt(gamma)-w >abx
    p[pos]=0
    return p


def prox_l1(x, w) :
    """
    Compute proximity operator of L1 norm"
    """
    p = np.zeros_like(x)
    pos = np.nonzero(x>w)[0]
    p[pos] = x[pos] - w
    neg = np.nonzero(x<-w)[0]
    p[neg] = x[neg] + w
    return p

def prox_l2(x, dx, eta) :
    """
    Compute projection of x onto l2 ball ||z-dx||<=eta
    x and dx are image vectors
    """
    t = x-dx
    s = t*np.minimum(eta/np.linalg.norm(t),1)
    return x + s - t

def lambert_w(x):
    """
    W Lambert function
    """
    d = scipy.special.lambertw(x, k=0, tol=1e-8)
    return np.real(d)

def approx_lambert(x):
    """
    approximation of W( exp(x) )
    no error below 50, and  less than 0.2% error for x>50 and converges toward 0 for x->inf
    does not overflow !
    """
    limit = 100
    nz = np.nonzero( (x < limit) )[0]
    A = 0.00303583748046
    s = x*(1 - np.log(x)/(1+x)) + A/(1+x-limit)**0.75
    s[nz] = lambert_w(np.exp(x[nz]))
    return np.nan_to_num(s)

def prox_l1_Sent(x, lamda, a):
    """
    Compute the proximity operator of L1 + Shannon Entropy
    """
    if lamda == 0:
        p = prox_l1(x, 1)
    else:
        loga = np.log(a)
        loglamda  = np.log(lamda)
        c = (a*x - a*(1-lamda))/lamda - 1 - loglamda + 2*loga
        p = (lamda/a)*approx_lambert(c)
    p[p<0]=0
    return p

def lipschitz_coeff(Kf, Ktf, N, nrandip=5):
    '''
    Lipfschitz coeff calculated from maximal eigenvalues
    N is the length of the observation
    nrandip is the number of trials
    '''
    rr = np.zeros((N,nrandip))
    for i in range(nrandip):              # sample N times the trf()
        r = np.random.randn(N)
        rr[:, i] = Ktf(Kf(r))             # go back and forth
    rrc = np.dot(np.conj(rr.T), rr)       # QQ*
    w, v = np.linalg.eig(rrc)             # computes the eigenvalues of the squared matrix
    w = np.sqrt(max(abs(w)))              # then computes largest eigenvalue of (trf o ttrf) 
    return 2*w

def norm2m(K, Kt, N, nbiter=50):
    """
    Computes the norm of K matrix
    """
    b = np.random.rand(N)
    for i in range(nbiter):
        tmp = np.dot(Kt,np.dot(K,b)) # Kt(K(b))   #tmp = (K^T * K) * b
        norm = np.sqrt(np.dot(tmp,tmp) )
        b = tmp/norm
    return np.linalg.norm(np.dot(K,b)) #np.sqrt(normb)

def norm2f(K, Kt, N, nbiter=50):
    """
    Computes the norm of Kf operator
    """
    b = np.random.rand(N)
    for i in range(nbiter):
        tmp = Kt(K(b)) # Kt(K(b))   #tmp = (K^T * K) * b
        norm = np.sqrt(np.dot(tmp,tmp) )
        b = tmp/norm
    return np.linalg.norm(K(b)) #np.sqrt(normb)


def pds_vf(Kf, Kft, N, y, eta, nbiter=100, normK=None, tau=None, lamda=0.0, prec= 1e-14, ref=None, Khi2=False, show=False):
    """
    This is a fonctional version of the Primal Dual algorithm
    - key indicate the choice of the regularization function Phi {l0, l1, logsum, etc}
    - Kf is the direct, Kft is the transpose
    - y is the experiment vector in data space
    - N is the size of the image (result) space
    - image -> Kf  -> data
    - data  -> Kft -> image
    This algorithm performs the Primal Dual Scheme:
    The optimization problem is : 
                                min_{x} f(x) + g(Lx) where L is a linear operator
    The dual problem is :         
                                min_{u} f^*(-L^*u) + g^*(u)
    In our case, f is the l_2 norme, L presents the measure operator <--> K, 
    and g is our regularization function <--> Psi = lamda entropy + (1-lamda) sparsity
    lamda controls the sparsity degree (lamda = 0 --> pure sparse)

    ref : if given, contains the real solution to the problem,
        and an array containing the distance in dB of the current solution to the real one is returned
        if None, and Khi2 is True, the returned array contains the normalized Khi2

    This is a fonctional version, Kf is the direct, Kft is the transpose
    """
    if normK is None:
        normK = norm2f(Kf, Kft, N)
        print ('normK',normK)
    if tau is None:
        tau = 1/normK
    sigma = 0.9/(tau*normK**2)  # sigma < 1/(tau*normK**2)
    ro = 1.99
    xk_old = np.zeros(N)
    uk_old = Kf(xk_old)
    refspec = []    
    cv_u = []
    cv_x = []
    noise = eta/np.sqrt(len(y))
    #key = 'l1'
    for k in range(0,nbiter): 
        if k%5 == 0:
            sys.stdout.write("\r %d iterations are done "%k)  
        xxk = prox_l1_Sent(xk_old - tau*(Kft(uk_old)), lamda=lamda, a=tau)
        zk = uk_old + sigma*(Kf(2*xxk-xk_old))
        uuk = zk - sigma*prox_l2(zk/sigma, y, eta) 
        xk = xk_old + ro*(xxk - xk_old)
        uk = uk_old + ro*(uuk - uk_old)
        ex = np.linalg.norm(xk-xk_old)**2 / np.linalg.norm(xk)**2
        eu = np.linalg.norm(uk-uk_old)**2 / np.linalg.norm(uk)**2
        cv_x.append(ex)
        cv_u.append(eu)
        #sparsity constraint
        #xk[xk<0] = 0.0
        #update
        uk_old = uk
        xk_old = xk
        #options
        if ref is not None:
            refspec.append(-10*np.log10(np.linalg.norm(xk.ravel()-ref.ravel())**2 / np.linalg.norm(ref.ravel())**2))
        else:
            if Khi2:
                refspec.append(sum((y-Kf(xk))**2)/eta)
        if ex < prec:
            print ("converged")
            break
    xk[xk<0] = 0.0
    refspec = np.array(refspec)
    print("\r %d iterations are done \n"%(k+1)) 

    #figures
    if show:
        print ("\n Recovered signal have the size =", xk.shape)
        print ("\n Measured  signal have the size =", y.shape)
        print ("\n K norm =", normK ," \n") 
        plt.plot(refspec, 'r-')
        plt.title("||xrec - xorig||/||xorig||")
        plt.xlabel("iterations")
        plt.ylabel("Intensity")
    print(len(xk), 'data points, ', len(refspec), 'iterations')
    return xk, refspec

def pds_vm(K, y, eta, nbiter=10, normK=None, tau=None, prec= 1e-14, ref=None, show=False):
    """
    This algorithm performs the Primal Dual Scheme:
    The optimization problem is : 
                                min_{x} f(x) + g(Lx) where L is a linear operator
    The dual problem is :         
                                min_{u} f^*(-L^*u) + g^*(u)
    In our case, f is the l_2 norme, L presents the measure operator <--> K, 
    and g is our regularization function <--> Psi = lamda entropy + (1-lamda) sparsity
    lamda controls the sparsity degree (lamda = 0 --> pure sparse)
    - key indicate the choice of the regularization function Phi {l0, l1, logsum, etc}
    - K is the direct matrix
    - y is the experiment vector in data space
    - N is the size of the image (result) space
    - image -> K  -> data
    - data  -> K.T -> image
    ref : if given, contains the real solution to the problem,
        and an array containing the distance in dB of the current solution to the real one is returned
        if None, and Khi2 is True, the returned array contains the normalized Khi2
    """
    M,N = K.shape
    if normK is None:
        normK = norm2m(K, K.T, N) #(lambda v: np.dot(K,v), lambda v: np.dot(K.T,v), K.shape[0]) #np.linalg.norm(K,2)
    if tau is None:
        tau = 1/normK
    sigma = 0.9/(tau*normK**2)  # sigma < 1/(tau*normK**2)
    ro = 1.99
    u0 = np.zeros_like(y)
    xk_old = np.zeros(N)
    x0 = xk_old.copy()
    uk_old = np.dot(K,x0)
    refspec = []
    for k in range(0,nbiter):
        sys.stdout.write("\r %d iterations are done"%k)
        xxk = prox_l1_Sent(xk_old - tau*np.dot(K.T, uk_old), lamda=0, a=tau)
        zk = uk_old + sigma*np.dot(K, 2*xxk-xk_old)
        uuk = zk - sigma*prox_l2(zk/sigma, y, eta) 
        xk = xk_old + ro*(xxk - xk_old)
        uk = uk_old + ro*(uuk - uk_old)
        ex = np.linalg.norm(xk-xk_old)**2 / np.linalg.norm(xk)**2
        eu = np.linalg.norm(uk-uk_old)**2 / np.linalg.norm(uk)**2
        if ex < prec and eu < prec:
            break
        #sparsity constraint
        #xk[xk <0] = 0 
        #update
        uk_old = uk
        xk_old = xk
        #options
        if ref is not None: 
            refspec.append(-10*np.log10(np.linalg.norm(xk.ravel()-ref.ravel())**2 / np.linalg.norm(ref.ravel())**2))
    #figures
    if show == True:
        print ("\n Recovered signal have the size =", xk.shape)
        print ("\n Measured  signal have the size =", y.shape)
        print ("\n Matrix K have the size =", K.shape)
        print ("\n K norm =", normK ," \n")
        plt.plot(refspec, 'r-')
        plt.title("||xrec - xorig||/||xorig||")
        plt.xlabel("iterations")
        plt.ylabel("Intensity")
    return xk, refspec
