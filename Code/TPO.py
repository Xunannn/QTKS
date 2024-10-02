# -*- coding: utf-8 -*
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mealpy import TPO
from objectfun import EnvelopeEntropyCost,SampleEntropyCost,PermutationEntropyCost,infoEntropyCost,compositeEntropyCost

fs = 12000
Ts = 1/fs
L = 4018
t = np.arange(0, L) * Ts
STA = 1

data1=pd.read_excel("VMD.xlsx",sheet_name='Sheet1')
data2=data1.to_numpy()
data = data2[STA-1:STA-1+L]
# print(data)


def x_EnvelopeEntropyCost(x):
    ff = EnvelopeEntropyCost(x,data)
    return ff
def x_SampleEntropyCost(x):
    ff = SampleEntropyCost(x,data)
    return ff
def x_PermutationEntropyCost(x):
    ff = PermutationEntropyCost(x,data)
    return ff
def x_infoEntropyCost(x):
    ff = infoEntropyCost(x,data)
    return ff
def x_compositeEntropyCost(x):
    ff = compositeEntropyCost(x,data)
    return ff

problem_dict = {
    "fit_func": x_PermutationEntropyCost,
    "lb": [100,5],
    "ub": [500,20],
    "minmax": "min",
}


epoch = 15
pop_size = 20

TPO_model = TPO.OriginalTPO(epoch, pop_size)
TPO_best_x, TPO_best_f = TPO_model.solve(problem_dict)
print('Optimum parameter：','alpha:',TPO_best_x[0],'K:',TPO_best_x[1],'Optimum fitness value：',TPO_best_f)

plt.figure()

x=np.arange(0,epoch)+1
x[0]=1
my_x_ticks = np.arange(1, epoch+1, 1)
plt.xticks(my_x_ticks)
plt.plot(x,TPO_model.history.list_global_best_fit ,'m-' ,linewidth=1 ,label = 'TPO',marker = "p",markersize=8)

plt.xlabel('Iteration')
plt.ylabel('Fitness')
plt.grid()
plt.legend()
plt.show()
