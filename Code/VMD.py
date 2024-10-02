import numpy as np
import pandas as pd
from matplotlib import font_manager
from vmdpy import VMD
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from Entropy_function import fuzzy_entropy

font_path = "黑体.TTF"

font_prop = font_manager.FontProperties(fname=font_path)

fs = 12000
Ts = 1.0/fs
L = 4018
t = np.arange(0, L) * Ts
STA = 1

excel_file_path = 'original data.xlsx'
df = pd.read_excel(excel_file_path, sheet_name='Sheet1', header= None)
f=np.array(df)

alpha = 50
tau = 0
K = 13
DC = 0
init = 2
tol = 1e-7

u,u_hat,omega = VMD(f, alpha=alpha, tau=tau, K=K, DC=DC, init=init, tol=tol)

df_modes = pd.DataFrame(index=range(len(u[0])))

for i in range(K):
    mode_i = pd.DataFrame(u[i].T)
    mode_i.columns = ['IMF{}'.format(i+1)]
    df_modes = pd.concat([df_modes, mode_i], axis=1)

print(df_modes)

d=u.T
a=[]
for i in range(len(d)):
    s=0
    for j in range(len(d[i])):
        s=s+d[i,j]
    a.append(s)
mse=mean_squared_error(f,a)
print(np.sqrt(mse),r2_score(f,a))

df_modes.to_excel('VMD_result.xlsx', index=False)

imfn = u
# 计算每个IMF的模糊熵
# 设置参数
r0 = 0.15  # r为相似容限度
K = len(imfn)  # 假设 imfn 是一个二维数组，每行是一个 IMF（本地变量中没有提到）

# 初始化模糊熵数组
FuEn = np.zeros(K)

# 计算每个IMF的模糊熵
for i1 in range(K):
    imf0 = imfn[i1, :]
    x = imf0
    r = r0 * np.std(x)
    FuEnx = fuzzy_entropy(x, 6, r, 2, 1)  # 模糊熵
    FuEn[i1] = FuEnx
    print(f'IMF{i1 + 1}的模糊熵为：{FuEn[i1]}')