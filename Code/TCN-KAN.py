import os
import warnings
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from prettytable import PrettyTable
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score
from math import sqrt
import argparse
import time
from src.model import *
warnings.filterwarnings("ignore")
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--vision', type=bool, default=True)
parser.add_argument('--train_test_ratio', type=float, default=0.2)
parser.add_argument('--random_state', type=int, default=34)
# model
parser.add_argument('--model_name', type=str, default='TCN_ekan')
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--hidden_dim', type=int, default=32)
parser.add_argument('--n_layers', type=int, default=2)
##kan
parser.add_argument('--grid_size', type=int, default=200,help='grid')
##TCN
parser.add_argument('--num_channels', type=list, default=[25, 50, 25])
parser.add_argument('--kernel_size', type=int, default=3)
##transformer
parser.add_argument('--num_heads', type=int, default=4)
parser.add_argument('--hidden_space', type=int, default=32)
# training
parser.add_argument('--num_epochs', type=int, default=300)
parser.add_argument('--seed', type=int, default=1)
# optimizer
parser.add_argument('--lr', type=float, default=5e-4, help='Adam learning rate')
args = parser.parse_args(args=[])
args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset=pd.read_excel('predict.xlsx')

print(dataset)#

values = dataset.values[:,2:]
print(values)

values = np.array(values)

num_samples = values.shape[0]
n_out = 1
per=np.arange(num_samples)
# print(per)
n_train_number = per[:int(num_samples * 0.8)]
n_test_number = per[int(num_samples * 0.8):]

Xtrain = values[n_train_number, :values.shape[1]-n_out]
Ytrain = values[n_train_number, values.shape[1]-n_out:]
Ytrain = Ytrain.reshape(-1,n_out)

Xtest = values[n_test_number, :values.shape[1]-n_out]
Ytest = values[n_test_number,  values.shape[1]-n_out:]
Ytest = Ytest.reshape(-1,n_out)

m_in = MinMaxScaler()
vp_train = m_in.fit_transform(Xtrain)
vp_test = m_in.transform(Xtest)
m_out = MinMaxScaler()
vt_train = m_out.fit_transform(Ytrain)
vt_test = m_out.transform(Ytest)


vp_train = vp_train.reshape((vp_train.shape[0], 1, vp_train.shape[1]))

vp_test = vp_test.reshape((vp_test.shape[0], 1, vp_test.shape[1]))

input_dim = vp_train.shape[2]
output_dim = n_out
hidden_dim = 1136
num_layers = 105


# 转换为torch数据

X_TRAIN = torch.from_numpy(vp_train).type(torch.Tensor)
Y_TRAIN = torch.from_numpy(vt_train).type(torch.Tensor)
X_TEST = torch.from_numpy(vp_test).type(torch.Tensor)
Y_TEST = torch.from_numpy(vt_test).type(torch.Tensor)

model=TemporalConvNet_ekan(num_inputs=input_dim, num_outputs=output_dim ,num_channels=args.num_channels, kernel_size=args.kernel_size, dropout=args.dropout)


criterion = torch.nn.MSELoss()
optimiser = torch.optim.Adam(model.parameters(), lr=0.01)

MSE_hist = np.zeros(args.num_epochs)
R2_hist = np.zeros(args.num_epochs)

start_time = time.time()
result = []

for t in range(args.num_epochs):
    y_train_pred = model(X_TRAIN)
    loss = criterion(y_train_pred, Y_TRAIN)
    R2 = r2_score(y_train_pred.detach().numpy(), Y_TRAIN.detach().numpy())
    print("Epoch ", t, "MSE: ", loss.item(), 'R2', R2.item())
    MSE_hist[t] = loss.item()
    if R2 < 0:
        R2 = 0
    R2_hist[t] = R2
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()
training_time = time.time() - start_time
print("Training time: {}".format(training_time))

y_test_pred = model(X_TEST)

yhat = y_test_pred.detach().numpy()
yhat = yhat.reshape(vp_test.shape[0], n_out)

predicted_data = m_out.inverse_transform(yhat)
print(predicted_data)
df = pd.DataFrame(predicted_data, columns=['Predicted_Value'])

df.to_excel(os.path.join('document path', 'result.xlsx'), index=False)

def mape(y_true, y_pred):
    record = []
    for index in range(len(y_true)):
        temp_mape = np.abs((y_pred[index] - y_true[index]) / y_true[index])
        record.append(temp_mape)
    return np.mean(record)

def NSE(observed, simulated):
    observed=np.array(observed)
    simulated=np.array(simulated)
    x_mean = simulated.mean()
    SST = np.sum((simulated - x_mean) ** 2)
    SSRes = np.sum((simulated - observed) ** 2)
    nse = 1 - (SSRes / SST)
    return nse


def evaluate_forecasts(Ytest, predicted_data, n_out):
    mse_dic = []
    rmse_dic = []
    mae_dic = []
    mape_dic = []
    r2_dic = []
    nse_dic=[]
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
        nse = NSE(actual, predicted)
        nse_dic.append(nse)
        if n_out == 1:
            strr = 'Prediction outcome index：'
        else:
            strr = 'No.'+ str(i + 1)
        table.add_row([strr, mse, rmse, mae, str(MApe)+'%', str(r2*100)+'%'])
    return mse_dic,rmse_dic, mae_dic, mape_dic, r2_dic, nse_dic ,table

mse_dic,rmse_dic, mae_dic, mape_dic, r2_dic, nse_dic , table = evaluate_forecasts(Ytest, predicted_data, n_out)

print(table)
print(nse_dic)

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
for ii in range(n_out):

    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(10, 2), dpi=300)
    x = range(1, len(predicted_data) + 1)
    plt.tick_params(labelsize=5)
    plt.plot(x, predicted_data[:,ii], linestyle="--",linewidth=0.8, label='predict',marker = "o",markersize=2)

    plt.plot(x, Ytest[:,ii], linestyle="-", linewidth=0.5,label='Real',marker = "x",markersize=2)

    plt.rcParams.update({'font.size': 5})

    plt.legend(loc='upper right', frameon=False)

    plt.xlabel("Sample points", fontsize=5)

    plt.ylabel("value", fontsize=5)

    if n_out == 1:
        plt.title(f"The prediction result of {args.model_name} :\nMAPE: {mape(Ytest[:, ii], predicted_data[:, ii])} %")
    else:
        plt.title(f"{ii+1} step of {args.model_name} prediction\nMAPE: {mape(Ytest[:,ii], predicted_data[:,ii])} %")

plt.ioff()
plt.show()