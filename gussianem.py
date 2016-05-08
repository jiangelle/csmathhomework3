# -*- coding: utf-8 -*-
"""
Created on Sun May 01 22:35:01 2016

@author: jurcol
"""
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la
import random
import math

mean1 = [0,0]
cov1 = [[1,0],[0,2]]
mean2 =[3,3]
cov2 = [[3,0],[0,2]]
num_of_data = 5000

plt.figure(0)
x = np.zeros((num_of_data,2))
for i in range(num_of_data):
    if random.uniform(0,1)>0.7:
        x[i,:] = np.random.multivariate_normal(mean1,cov1,1)
        plt.plot(x[i,0],x[i,1],'.',color = "green")
    else :
        x[i,:] = np.random.multivariate_normal(mean2,cov2,1)
        plt.plot(x[i,0],x[i,1],'.',color = "red")

#print x
def computegussian(x,u,sigm):
    return 1/(2*math.pi*math.sqrt(np.linalg.det(sigm)))*math.exp(-1.0/2*np.dot(np.dot((x-u),np.linalg.inv(sigm)),(x-u).T))

#init the mean , covariance, weight
u = np.array([[0.1,0.2],[0.2,0.3]])
sigm = np.array([[0.5,0],[0,0.6],[0.4,0],[0,0.3]])
pai = np.array( [0.4,0.6])

while True:
    uold = u.copy()
    sigmold = sigm.copy()
    paiold = pai.copy()
    #compute the density of point
    N = np.zeros([num_of_data,2])
    for m in range(num_of_data):
        for n in range(2):
            N[m,n]= computegussian(x[m,:],u[n,:],sigm[n*2:n*2+2,:])

    #compute the  E-step      
    sumpoint = np.zeros(num_of_data)
    for m in range(num_of_data):
        for n in range(2):
            sumpoint[m] += pai[n]*N[m,n]
       
    gamma = np.zeros([num_of_data,2])
    for m in range(num_of_data):
        for n in range(2):
            gamma[m,n] = pai[n]*N[m,n]/sumpoint[m]
    #compute the M-step(max log likelihood)
    C = np.zeros(2)
    for n in range(2):
        C[n] = np.sum(gamma[:,n])
        pai[n] = C[n]/num_of_data
        u[n,:] = np.dot(gamma[:,n].T,x)/C[n]
    for n in range(2):
        S = np.zeros((2,2))
        for m in range(num_of_data):
            S += gamma[m,n]*np.dot((x[m,:]-u[n,:]).reshape(2,1),(x[m,:]-u[n,:]).reshape(1,2))
        sigm[n*2:n*2+2,:] = S/C[n]
    print la.norm(u - uold, 2)
    if la.norm(u - uold, 2) < 1e-9 and la.norm(sigm - sigmold, 2) < 1e-9 and la.norm(pai - paiold, 2) < 1e-9:
        break

print u
print sigm
print pai

plt.figure(1)
for i in range(num_of_data):
    if random.uniform(0,1)>pai[0]:
        temp = np.random.multivariate_normal(u[0,:],sigm[0:2,:],1)
        plt.plot(temp[0,0],temp[0,1],'.',color = "blue")
    else :
        temp = np.random.multivariate_normal(u[1,:],sigm[2:4,:],1)
        plt.plot(temp[0,0],temp[0,1],'.',color = "yellow")

plt.show()