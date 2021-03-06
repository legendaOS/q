#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import math
import pandas
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, SimpleRNN, LSTM, Flatten
from sklearn.cluster import DBSCAN, KMeans
from sklearn import metrics
import matplotlib.pyplot as plt


# In[2]:


def mmean(arr):
    sr = 0
    c = 0
    for i in arr:
        sr += i
        c += 1
    return sr/c


# In[3]:


def trans_to_softmax(val, shape):
    buf = [0 for i in range(shape)]
    buf[val] = 1
    return buf


# In[4]:


from random import randint as randint
def randcolor(n):
    buf = [str('#' + str(randint(10,99)) + '' + str(randint(10,99)) + ''+ str(randint(10,99))) for i in range(n)]
    return buf


# Создание выборки по углам

# In[5]:


def create_tg_viborka(arr):
    tg_a_p = []
    for i in arr:
        buf = []
        for c in range(len(i) - 1):
            buf.append(math.atan( ( i[c] - i[len(i)-1] ) / (len(i) - 1) ) )
        tg_a_p.append(buf)
    return tg_a_p


# Создание выборки для кластеризации

# In[6]:


def create_viborka(start, arr):
    arr_sr_t = []
    for i in range(start, len(arr)+1):
        arr_sr_t.append(arr[i - start:i])
    return arr_sr_t


# Визуализация результатов класстеризации

# In[7]:


def visualize(n_clust, predictions, arr, file_name = 'ind.png', sh = False):

    colors = randcolor(n_clust)

    not_train = len(arr) - len(predictions)
    
    
    for i in range(len(predictions)):
        color = predictions[i]
        val = arr[i + not_train]
        plt.scatter(i + not_train, val, c = colors[color])
        
    plt.plot(arr)
    if(sh):
        plt.show()
    plt.savefig(file_name)
    


# Чиатем файл 2018 года и составляем список температур

# In[8]:


d = pandas.read_excel('2018.xlsx', engine='openpyxl')
temp = []
buf = []
c = 0
for i in d['T']:
  buf.append(i)
  c += 1
  if c == 8:
    temp.append(buf)
    buf = []
    c = 0


# Список средних значений в пачке

# In[9]:


def mean_arr(arr):
    sr_temp = []
    for i in arr:
        sr_temp.append(mmean(i))
    return sr_temp


# In[10]:


sr_temp = mean_arr(temp)


# Выборка для класстеризации, количестов кластеров, сколько элементов в пачке для выборки

# In[11]:


clusters__ = 8
in_vibor__ = 6

x_train = create_viborka(in_vibor__, sr_temp)
x_tg_train = create_tg_viborka(x_train)


# In[87]:


x_tg_train


# Создание модели для класстеризации, ее обучение и прогнозировние

# In[12]:


len(sr_temp)


# In[13]:


k_mean_model = KMeans(n_clusters = clusters__)
k_mean_model.fit(x_tg_train)
predictions = k_mean_model.predict(x_tg_train)


# In[14]:


visualize(clusters__,predictions,  sr_temp , sh = True)


# Модель определяющая какой из кластеров будет после предыдущих N кластеров

# Формирование обучающей выборки для модели предсказания кластера

# In[15]:


size_of_vib = in_vibor__ - 1

x_train_cluster = []
y_train_cluster = []
for i in range(size_of_vib, len(predictions)):
    y_train_cluster.append(trans_to_softmax(predictions[i], clusters__))


for i in range(len(predictions) - size_of_vib):
    buf = [] 
    for j in range(i, i+size_of_vib):
        buf.append(float(predictions[j]))
    x_train_cluster.append(buf)


# In[16]:


x_train_cluster


# Модель

# In[17]:


model = keras.Sequential()
model.add(Dense(input_dim = size_of_vib, units=80, activation='tanh')) 
model.add(Dense(units=128, activation='tanh'))
model.add(Dense(units=128, activation='tanh'))
model.add(Dense(units = clusters__, activation='softmax'))

model.compile(optimizer='RMSProp', loss='categorical_crossentropy', metrics=['accuracy'])


# In[18]:


info = model.fit(x_train_cluster, y_train_cluster, epochs = 300)


# In[19]:


predictions


# In[20]:


np.argmax(model.predict([[6,0,6,6,1]]))


# Создаем для каждого кластера свою сеть для предсказания следущего значения

# In[21]:


NN = []
for i in range(clusters__):
    NN.append(keras.Sequential())


# In[22]:


NN


# Подготовка нсетей

# In[23]:


for ns in NN:
    ns.add(Dense(input_dim = size_of_vib, units = 100, activation='linear'))
    ns.add(Dense(units = 60, activation='linear'))
    ns.add(Dense(units = 60, activation='linear'))
    ns.add(Dense(units = 60, activation='linear'))
    ns.add(Dense(units = 1))
    ns.compile(loss = 'mae', optimizer = 'adam')


# Формирование обучающих и тестовых выборок

# In[24]:


x_last_train = [[] for i in range(clusters__)]
y_last_train = [[] for i in range(clusters__)]


# In[25]:


for i in range(len(sr_temp) - in_vibor__):
    buf = sr_temp[i:i+size_of_vib]
    to_trans_buf = sr_temp[i:i+in_vibor__]
    tg_v = create_tg_viborka([to_trans_buf])
    predict_on = k_mean_model.predict(tg_v)
    pre = int(predict_on[0])
    yy = [to_trans_buf[-1]]
    

    x_last_train[pre].append(buf)
    y_last_train[pre].append(yy)


# In[26]:


for n in range(len(NN)):
    NN[n].fit(x_last_train[n], y_last_train[n], epochs = 200)


# In[27]:


sr_temp


# In[83]:


print(predictions)
ptest = NN[1].predict([[-2.4499999999999997,-3.4625000000000004,-6.675000000000001,-6.5625,-7.8125,]])
print(ptest)
print(sr_temp[5])


# In[74]:


print('ошибка ' + str(float((sr_temp[5] - ptest)/sr_temp[5]) * 100) + ' %')


# Что бы увеличить точность нужно подгонять кол-во кластеров и дней в выборке
# полет фантазии

# In[61]:


d_n = pandas.read_excel('2019.xlsx', engine='openpyxl')
temp_n = []
buf_n = []
c_n = 0
for i in d_n['T']:
  buf_n.append(i)
  c_n += 1
  if c_n == 8:
    temp_n.append(buf_n)
    buf_n = []
    c_n = 0


# In[62]:


sr_temp_n = mean_arr(temp_n)
sr_temp_n


# In[63]:


x_train_n = create_viborka(in_vibor__, sr_temp_n)
x_tg_train_n = create_tg_viborka(x_train_n)


# In[64]:


x_tg_train_n


# In[65]:


predictions_n = k_mean_model.predict(x_tg_train_n)


# In[82]:


print(predictions_n)
ptest_n = NN[0].predict([[0.9499999999999998,-2.9,-3.4499999999999997,-1.5,-0.65]])
print(ptest_n)
print(sr_temp_n[5])

