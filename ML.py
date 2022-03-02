Main function():

Created on Thu Jun 18 17:35:07 2020

@author: zhewang
"""

import numpy as np
from pore_parser import pore_shower
from xyratio import orientation
from break_energy import energy
from disperse import disperse
from math import sqrt
import pandas as pd

import glob
import re
numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def normalization(somelist):
    listmax=max(somelist)
    listmin=min(somelist)
    listspan=listmax-listmin
    for i in range(len(somelist)):
        somelist[i]=(somelist[i]-listmin)/listspan
    return somelist

gridpath='/Users/zhewang/Documents/spyder/cee298final/grids/grid.*.csv'
stresspath='/Users/zhewang/Documents/spyder/cee298final/curves/stress.*.csv'
strainpath='/Users/zhewang/Documents/spyder/cee298final/curves/strain.csv'
grids=sorted(glob.glob(gridpath), key=numericalSort)
stresses=sorted(glob.glob(stresspath), key=numericalSort)


    
break_energy=[]
for stress in stresses:
    break_energy.append(energy(strainpath,stress))

num_feat=3
porosity=[]
xyratio=[]
dispersity=[]
for grid in grids:
    pore_map=pore_shower(grid)
    xyratio.append(orientation(pore_map))
    dispersity.append(disperse(pore_map))
    
    pores=0
    for pore in pore_map:
        pores=pores+len(pore)
    porosity.append(pores)
    
    #more features...

porosity=normalization(porosity)
xyratio=normalization(xyratio)
dispersity=normalization(dispersity)
X=list(zip(porosity,xyratio,dispersity))#+other nomarlized features

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, break_energy, random_state=0)


# from sklearn.linear_model import LinearRegression
# linreg=LinearRegression(normalize=True)
# linreg.fit(x_train,y_train)
# y_pred=linreg.predict(x_test)
# print('Score of linear regression: {:.2f}'
#      .format(linreg.score(x_test, y_test)))

from sklearn.linear_model import Lasso
def lasso_regression(x_train, y_train, x_test, y_test, alpha):
    #Fit the model
    lassoreg = Lasso(alpha=alpha,normalize=True, max_iter=1e5)
    lassoreg.fit(x_train,y_train)
    y_pred = lassoreg.predict(x_test)
    #Return the result in pre-defined format
    rmse = sqrt(np.mean((y_pred-y_test)**2)) #recording data and print
    ret = [rmse]
    ret.extend([lassoreg.intercept_])
    ret.extend(lassoreg.coef_)
    ret.extend([lassoreg.score(x_test,y_test)])
    return ret

alpha = [1e-10,1e-9,1e-7,1e-5,1e-3,0.01,0.1] #the trial alpha parameter
#Initialize the dataframe to store coefficients
col = ['rmse','intercept'] + ['porosity']+['orientation']+['dispersity']+['score']
ind = ['alpha_%.2g'%alpha[i] for i in range(len(alpha))]
coef_matrix_lasso = pd.DataFrame(index=ind, columns=col) #the table of coefficients

#Iterate over the 10 alpha values:
for i in range(len(alpha)):
    coef_matrix_lasso.iloc[i,] = lasso_regression(x_train, y_train, x_test, y_test, alpha[i]) #driver function

Helper functions:

import pandas as pd

def energy(strainsource,stresssource):
    strainlist=(pd.read_csv(strainsource,header=None)).values.tolist()
    stresslist=(pd.read_csv(stresssource,header=None)).values.tolist()
    strain=[strainlist[i][0] for i in range(len(strainlist))]
    stress=[stresslist[i][0] for i in range(len(stresslist))]
    ss_map=[]
    ss_map.append([strain[0],0])
    for i in range(len(stress)):
        ss_map.append([strain[i+1],stress[i]])
    ss_map.append([strain[len(strain)-1],0])
    ss_map=sorted(ss_map,key=lambda x:x[0])
    energy=0
    for i in range(len(ss_map)-1):
        area=(ss_map[i][1]+ss_map[i+1][1])*(ss_map[i+1][0]-ss_map[i][0])/2
        energy=energy+area
    return energy

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

types = ['connected', 'border', 'isolated']

def safe_get(grid, i, j):
    if i < 0 or i > grid.shape[0] - 1 or j < 0 or j > grid.shape[1] - 1:
        return -1
    return grid[i, j]

def is_connected(grid, i, j):
    return safe_get(grid, i - 1, j) == 1 or safe_get(grid, i + 1, j) == 1 or safe_get(grid, i, j - 1) == 1 or safe_get(grid, i, j + 1) == 1

def is_border(grid, i, j):
    return safe_get(grid, i - 1, j - 1) == 1 or safe_get(grid, i - 1, j + 1) == 1 or safe_get(grid, i + 1, j - 1) == 1 or safe_get(grid, i + 1, j + 1) == 1

def get_pores(grid, i, j, cur_type):
    res = []
    if cur_type == 2:
        return res
    
    straight = [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]
    # diagnol = [(i - 1, j - 1), (i - 1, j + 1), (i + 1, j - 1), (i + 1, j + 1)]

    if cur_type == 0:
        for (m, n) in straight:
            val = safe_get(grid, m, n)
            if val == 1:
                res.append((m, n))
    # else:
    #     for (m, n) in diagnol:
    #         val = safe_get(grid, m, n)
    #         if val == 1:
    #             res.append((m, n))
    return res

def dfs(grid, i, j, visited, res, type, curr):
    curr.append((i, j))
    if is_connected(grid, i, j):
        cur_type = 0
    elif is_border(grid, i, j):
        cur_type = 1
    else:
        cur_type = 2
    
    if not type is None:
        if cur_type < type:
            return 
        cur_type = type

    visited[i][j] = True
    
    pores_list = get_pores(grid, i, j, cur_type)

    for (next_i, next_j) in pores_list:
        if not visited[next_i][next_j]:
            dfs(grid, next_i, next_j, visited, res, cur_type, curr)
    
    if type is None:
        res.append(curr)
        curr = [] 
        # columns=['type', 'volume', 'circularity'])

    return

#path='/Users/zhewang/Documents/spyder/cee298final/grids/grid.1.csv'

def pore_shower(path):
    #if __name__ == "__main__":
    grid = np.loadtxt(path, delimiter=",")
    n = grid.shape[0]
    visited = [[True if grid[i, j] == 0 else False for j in range(n)] for i in range(n)]
    res = []

    for i in range(n):
        for j in range(n):
            if visited[i][j]:
                continue
            dfs(grid, i, j, visited, res, None, [])
return res


import numpy as np
from math import sqrt

def findcentroid(somelist):
    Mx=np.mean([somelist[i][0] for i in range(len(somelist))])
    My=np.mean([somelist[i][1] for i in range(len(somelist))])
    M=[Mx,My]
    return M

def disperse(pores):
    centroids=[]
    poresize=[]
    for pore in pores:
        centroids.append(findcentroid(pore))
        poresize.append(len(pore))
    Mx=np.sum([centroids[i][0]*poresize[i] for i in range(len(pores))])/np.sum(poresize)
    My=np.sum([centroids[i][1]*poresize[i] for i in range(len(pores))])/np.sum(poresize)
    M=[Mx,My]
    dispersity=[]
    for centroid in centroids:
        dispersity.append(sqrt((centroid[0]-M[0])**2+(centroid[1]-M[1])**2))
    return np.mean(dispersity)


def orientation(island):
    porosity=0
    xy=0
    for pore in island:
        area=len(pore)
        xmax=max([pore[i][0] for i in range(len(pore))])
        xmin=min([pore[i][0] for i in range(len(pore))])
        ymax=max([pore[i][1] for i in range(len(pore))])
        ymin=min([pore[i][1] for i in range(len(pore))])
        xyr=(xmax-xmin+1)/(ymax-ymin+1)
        porosity=porosity+area
        xy=xy+xyr*area
    xyratio=xy/porosity
    return xyratio
