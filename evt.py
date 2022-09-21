from random import sample
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import scipy.stats 

class MyRandomVariableClass(stats.rv_continuous):
    # def __init__(self,  zeta=0.1, sigma=1, seed=None):
    def __init__(self,  seed=None):
        super().__init__(a=0,  seed=seed)

    # def _pdf(self, x):
        
    #     zeta=0.1
    #     sigma=1
    #     return 1/sigma*(1+zeta/sigma*x)**(-(zeta+1)/zeta)
        
    # def _cdf(self, x):
    def _cdf(self, x, zeta, sigma):
        # return 1-np.exp(-x**2)
        # return 1/sigma*(1+zeta/sigma*x)**(-(zeta+1)/zeta)
        return 1-(1+zeta/sigma*x)**(-1/zeta)

if __name__ == "__main__":
    my_rv = MyRandomVariableClass()

    # get sample_m from the given distribution
    sample_size_m=700
    zeta_sample_m=0.03
    sigma_sample_m=0.05
    samples_m = my_rv.rvs(size = sample_size_m, zeta=zeta_sample_m, sigma=sigma_sample_m)
    # samples[samples>1]=1
    # samples=np.where(samples>1)
    # print(samples)
    # sample_ub=1
    # samples=samples[samples<sample_ub]
    # print(len(samples_m))
    # print(samples_m)
    
    # get sample_mn from the given distribution



    # create a new distribution and use MLE to fit the samples
    fit_rv=MyRandomVariableClass()
    # print(fit_rv.fit(samples))
    # print(scipy.stats.fit(fit_rv, samples))
    # print(fit_rv.shapes)
    zeta_fit_m, sigma_fit_m, loc1, scale1=(fit_rv.fit(samples_m, floc=0, fscale=1)) #default method: MLE ,method='MLE'
    print(zeta_fit_m, sigma_fit_m, loc1, scale1)


    # plot histogram of samples
    fig, ax1 = plt.subplots()
    ax1.hist(list(samples_m),bins=50)

    # plot PDF and CDF of distribution
    pts = np.linspace(0, 10, 500)
    ax2 = ax1.twinx()
    ax2.set_ylim(0,1.1)
    ax2.plot(pts, my_rv.pdf(pts, zeta=zeta_sample_m, sigma=sigma_sample_m), color='red')
    ax2.plot(pts, my_rv.cdf(pts, zeta=zeta_sample_m, sigma=sigma_sample_m), color='orange')
    
    
    ax2.plot(pts, fit_rv.pdf(pts,zeta=zeta_fit_m,sigma=sigma_fit_m), color='navy')


    fig.tight_layout()
    plt.show()