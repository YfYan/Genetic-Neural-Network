#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 17 12:40:20 2020

@author: yanyifan
"""

import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

"""
神经网络的超参数
"""
MAX_HIDDEN_LAYER_NUM = 4
hidden_neurons = [4,8,16,32]
activations = ['relu','linear','sigmoid','elu']
optimizers = ['adam','rmsprop','sgd','adagrad']

"""
遗传算法的超参数
"""
POP_SIZE = 30
DNA_SIZE = 20
N_GENERATIONS = 50
CROSS_RATE = 0.8
MUTATION_RATE = 0.01


"""
数据读取 获得输入输出 分割训练测试集
"""
df = pd.read_csv('close.csv',encoding = 'gb2312')
data = list(df['AU100'])


def split_sequence(sequence, n_steps):
    """
    分割获得数据集
    """
    X, y = [], []
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence)-1:
          break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

step = 32
X,y = split_sequence(data,step)
y = y.reshape((-1,1))

train_num = int(len(y)*0.6)

X_train,y_train = X[:train_num,:],y[:train_num]
X_test,y_test = X[train_num:,:],y[train_num:]

X_scale = MinMaxScaler()
X_train = X_scale.fit_transform(X_train)
X_test = X_scale.transform(X_test)

#对数据做0-1归一化
y_scale = MinMaxScaler()
y_train = y_scale.fit_transform(y_train)


def binary2int(binary:list):
    """
    2进制list转变为整数
    """
    res = 0
    for i in range(len(binary)-1,-1,-1):
        res = res*2 + binary[i]
    return res

def decodeDNA(gene:list):
    """
    将DNA解码成神经网络结构
    Parameters
    ----------
    gene : list
        DESCRIPTION. DNA列表

    Returns
    -------
    model : TYPE keras.Sequential()
        DESCRIPTION. 搭好的神经网络结构

    """
    hidden_layer_num = binary2int(gene[:2]) + 1
    config = gene[2:2+hidden_layer_num*4]
    model = keras.Sequential()
    for layer in range(hidden_layer_num):
        act = activations[binary2int(config[layer*4:layer*4+2])]
        neuron_num = hidden_neurons[binary2int(config[layer*4+2:layer*4+4])]
        if layer == 0:
            model.add(keras.layers.Dense(neuron_num,activation = act,input_shape = [step]))
        else:
            model.add(keras.layers.Dense(neuron_num,activation = act,input_shape = [step]))
    model.add(keras.layers.Dense(1))
    opt = optimizers[binary2int(gene[-2:])]
    model.compile(loss = 'mse',optimizer = opt,metrics = ['mse'])
    return model

def decodeDNA_show(gene:list):
    """
    用于展示（print）某个DNA的模型
    """
    hidden_layer_num = binary2int(gene[:2]) + 1
    print(hidden_layer_num)
    config = gene[2:2+hidden_layer_num*4]
    
    for layer in range(hidden_layer_num):
        act = activations[binary2int(config[layer*4:layer*4+2])]
        neuron_num = hidden_neurons[binary2int(config[layer*4+2:layer*4+4])]
        print(neuron_num,act)   
    opt = optimizers[binary2int(gene[-2:])]
    print(opt)
    

def pop_test_mse(pop):
    """
    计算种群中各个神经网络的mse和预测结果

    Parameters
    ----------
    pop : TYPE np.ndarray
        DESCRIPTION. 种群DNA

    Returns
    -------
    TYPE np.ndarray
        DESCRIPTION. 各个DNA的mse
    predictions : TYPE np.ndarray
        DESCRIPTION. 各个DNA对应神经网络的预测结果

    """
    res = []
    predictions = []
    for index in range(POP_SIZE):
        cur_res = 10000
        cur_pred = None
        
        # 对每个DNA连续优化三次，取最低的MSE作为最好的结果。
        for cnt in range(3):
            cur = list(pop[index])
            model = decodeDNA(cur)
            early_stopping = EarlyStopping(monitor='val_loss', patience=50, verbose=2)
            model.fit(X_train,y_train,epochs=100, batch_size=80, validation_split = 0.3, verbose=1,callbacks = [ early_stopping])
            test_mse = np.mean(np.square(y_scale.inverse_transform(model.predict(X_test)) - y_test))
            # test_mse = np.mean(np.square(model.predict(X_test) - y_test))
            
            if np.isnan(test_mse) == True:
                test_mse = 100000  
    
            if test_mse < cur_res:
                cur_res = test_mse
                cur_pred = y_scale.inverse_transform(model.predict(X_test))
                # cur_pred = model.predict(X_test)
         
        
        predictions.append(cur_pred)
    
        res.append(test_mse)
    return np.array(res),predictions

def fitness(pop_test_mse):
    """
    适应度函数
    """
    return np.exp(-0.05*pop_test_mse**2)


def select(pop, fitness):
    """
    根据适应度的选择操作
    """
    idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True,
                           p=fitness/fitness.sum()) # p 就是选它的比例
    return pop[idx]

def crossover(parent, pop):
    """
    遗传算法的交叉操作
    """
    if np.random.rand() < CROSS_RATE:
        i_ = np.random.randint(0, POP_SIZE, size=1)  # select another individual from pop
        cross_points = np.random.randint(0, 2, DNA_SIZE).astype(np.bool)  # choose crossover points
        parent[cross_points] = pop[i_, cross_points]  # mating and produce one child
    return parent

def mutate(child):
    """
    对某个基因的突变操作
    """
    for point in range(DNA_SIZE):
        if np.random.rand() < MUTATION_RATE:
            child[point] = 1 if child[point] == 0 else 0
    return child

def discretion(pop):
    """
    种群离散程度的度量
    """
    m,n = pop.shape
    s = 0
    for i in range(m):
        for j in range(m):
            local = np.sum(np.abs(pop[i] - pop[j]))
            s += local
    return s/(m**2)

        
if __name__ == '__main__':
    pop = np.random.randint(2, size=(POP_SIZE, DNA_SIZE))
    mses = []
    discretions = []
    
    """
    遗传算法优化部分
    """
    for _ in range(N_GENERATIONS):
        discretions.append(discretion(pop))
        test_mse,predictions = pop_test_mse(pop)
        print(test_mse)
        pop_fitness = fitness(test_mse)
        print("Most fitted DNA: ", pop[np.argmax(pop_fitness), :])
        print("Structure:",decodeDNA_show(list(pop[np.argmax(pop_fitness), :])))
        best_predict = predictions[np.argmax(pop_fitness)]
        pop = select(pop, pop_fitness)
        pop_copy = pop.copy()
        if _ != N_GENERATIONS - 1:
            for parent in pop:
                child = crossover(parent, pop_copy)
                child = mutate(child)
                parent[:] = child
        mses.append(np.mean(test_mse))
    
    """
    遗传算法收敛性画图
    """
    plt.figure(figsize = (6,3),dpi = 224)
    plt.plot(range(1,N_GENERATIONS+1),mses,c = 'blue',marker = 'o',linewidth = 0.6,ms = 2)
    plt.xlabel('iterations')
    plt.ylabel('average mses on test set')
    plt.savefig("mses.png")
    plt.legend()
    plt.show()
    
    plt.figure(figsize = (6,3),dpi = 224)
    plt.plot(range(1,N_GENERATIONS+1),discretions,c = 'red',marker = 'o',linewidth = 0.6,ms = 2)
    plt.xlabel('iterations')
    plt.ylabel('discretion on population')
    plt.savefig("discretes.png")
    plt.legend()
    plt.show()
    
    
    
            
