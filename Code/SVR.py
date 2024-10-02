import os  # 导入os模块，用于操作系统功能，比如环境变量
import math  # 导入math模块，提供基本的数学功能
import pandas as pd  # 导入pandas模块，用于数据处理和分析
import openpyxl
from math import sqrt  # 从math模块导入sqrt函数，用于计算平方根
from numpy import concatenate  # 从numpy模块导入concatenate函数，用于数组拼接
import matplotlib.pyplot as plt  # 导入matplotlib.pyplot模块，用于绘图
import numpy as np  # 导入numpy模块，用于数值计算
# import tensorflow as tf  # 导入tensorflow模块，用于深度学习
from sklearn.preprocessing import MinMaxScaler  # 导入sklearn中的MinMaxScaler，用于特征缩放
from sklearn.preprocessing import StandardScaler  # 导入sklearn中的StandardScaler，用于特征标准化
from sklearn.preprocessing import LabelEncoder  # 导入sklearn中的LabelEncoder，用于标签编码
from sklearn.metrics import mean_squared_error  # 导入sklearn中的mean_squared_error，用于计算均方误差
from tensorflow.keras.layers import *  # 从tensorflow.keras.layers导入所有层，用于构建神经网络
from tensorflow.keras.models import *  # 从tensorflow.keras.models导入所有模型，用于构建和管理模型
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score  # 导入额外的评估指标
from pandas import DataFrame  # 从pandas导入DataFrame，用于创建和操作数据表
from pandas import concat  # 从pandas导入concat函数，用于DataFrame的拼接
import keras.backend as K  # 导入keras的后端接口
from scipy.io import savemat, loadmat  # 从scipy.io导入savemat和loadmat，用于MATLAB文件的读写
from sklearn.neural_network import MLPRegressor  # 从sklearn.neural_network导入MLPRegressor，用于创建多层感知器回归模型
from keras.callbacks import LearningRateScheduler  # 从keras.callbacks导入LearningRateScheduler，用于调整学习率
from tensorflow.keras import Input, Model, Sequential  # 从tensorflow.keras导入Input, Model和Sequential，用于模型构建
import mplcyberpunk
from qbstyles import mpl_style
from sklearn.svm import SVR
import warnings
from prettytable import PrettyTable #可以优美的打印表格结果
warnings.filterwarnings("ignore")
dataset=pd.read_excel("Test result.xlsx")
print(dataset)

values = dataset.values[:,2:]
values = np.array(values)
num_samples = values.shape[0]
per=np.arange(num_samples)
n_train_number = per[:int(num_samples * 0.8)]
n_test_number = per[int(num_samples * 0.8):]
Xtrain = values[n_train_number, :-1]
Ytrain = values[n_train_number, -1]
Ytrain = Ytrain.reshape(-1,1)
Xtest = values[n_test_number, :-1]
Ytest = values[n_test_number,  -1]
Ytest = Ytest.reshape(-1,1)

m_in = MinMaxScaler()
vp_train = m_in.fit_transform(Xtrain)
vp_test = m_in.transform(Xtest)
m_out = MinMaxScaler()
vt_train = m_out.fit_transform(Ytrain)
vt_test = m_out.transform(Ytest)

model = SVR(C=15, epsilon=0.01, gamma='auto')
model.fit(vp_train, vt_train)
yhat = model.predict(vp_test)
yhat = yhat.reshape(-1, 1)

predicted_data = m_out.inverse_transform(yhat)

print(predicted_data)
df=pd.DataFrame(predicted_data, columns=['Test result'])
df.to_excel(os.path.join('path',"excel.xlsx"))

def mape(y_true, y_pred):
    record = []
    for index in range(len(y_true)):
        temp_mape = np.abs((y_pred[index] - y_true[index]) / y_true[index])
        record.append(temp_mape)
    return np.mean(record) * 100

def evaluate_forecasts(Ytest, predicted_data, n_out):
    mse_dic = []
    rmse_dic = []
    mae_dic = []
    mape_dic = []
    r2_dic = []
    table = PrettyTable(['Test set pointer','MSE', 'RMSE', 'MAE', 'MAPE','R2'])
    for i in range(n_out):
        actual = [float(row[i]) for row in Ytest]
        predicted = [float(row[i]) for row in predicted_data]
        mse = mean_squared_error(actual, predicted)
        mse_dic.append(mse)
        rmse = sqrt(mean_squared_error(actual, predicted))
        rmse_dic.append(rmse)
        mae = mean_absolute_error(actual, predicted)
        mae_dic.append(mae)
        MApe = mape(actual, predicted)
        mape_dic.append(MApe)
        r2 = r2_score(actual, predicted)
        r2_dic.append(r2)
        if n_out == 1:
            strr = 'Prediction outcome index：'
        else:
            strr = 'No.'+ str(i + 1)
        table.add_row([strr, mse, rmse, mae, str(MApe)+'%', str(r2*100)+'%'])
    return mse_dic,rmse_dic, mae_dic, mape_dic, r2_dic, table

mse_dic,rmse_dic, mae_dic, mape_dic, r2_dic, table = evaluate_forecasts(Ytest, predicted_data, 1)

print(table)

from matplotlib import rcParams

config = {
            "font.family": 'serif',
            "font.size": 10,
            "mathtext.fontset": 'stix',
            "font.serif": ['Times New Roman'],
            'axes.unicode_minus': False
         }
rcParams.update(config)
plt.ion()


plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(8, 2), dpi=300)
x = range(1, len(predicted_data) + 1)
plt.tick_params(labelsize=5)
plt.plot(x, predicted_data, linestyle="--",linewidth=0.8, label='predict',marker = "o",markersize=2)
plt.plot(x, Ytest, linestyle="-", linewidth=0.5,label='Real',marker = "x",markersize=2)
plt.rcParams.update({'font.size': 5})
plt.legend(loc='upper right', frameon=False)
plt.xlabel("Sample points", fontsize=5)
plt.ylabel("value", fontsize=5)
plt.title(f"The prediction result of bagging :\nMAPE: {mape(Ytest, predicted_data)} %",fontsize=5)

plt.ioff()
plt.show()