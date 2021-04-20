#подключение библиотек

import numpy as np
import math
import pandas
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, SimpleRNN, LSTM, Flatten
from sklearn.cluster import DBSCAN, KMeans
from sklearn import metrics
import matplotlib.pyplot as plt
import pickle

#предопределенные переменные

filename = 'kmeans.sav'
clusters__ = 8
in_vibor__ = 6
size_of_vib = in_vibor__ - 1

#функции предсказания

def num_max(arr):
    buf = arr[0]
    index = 0
    for i in range(1, len(arr)):
        if arr[i] > buf:
            buf = arr[i]
            index = i
    return(index)

def create_tg_viborka(arr):
    tg_a_p = []
    for i in arr:
        buf = []
        for c in range(len(i) - 1):
            buf.append(math.atan( ( i[c] - i[len(i)-1] ) / (len(i) - 1) ) )
        tg_a_p.append(buf)
    return tg_a_p

def create_viborka(start, arr):
    arr_sr_t = []
    for i in range(start, len(arr)+1):
        arr_sr_t.append(arr[i - start:i])
    return arr_sr_t

def predict_one_day(temperatures, k_mean_model, nn_predict_cluster, nns_predicts_temp, size_of_vibor):
    if len(temperatures) != 2*size_of_vibor:
        return None
    
    temp_to_prognos = temperatures[len(temperatures) - size_of_vibor:len(temperatures)]
    
    in_vibor_ = size_of_vibor + 1
    tg_viborka = []
    x = create_viborka(in_vibor_, temperatures)
    tg_viborka = create_tg_viborka(x)
    
    pred = []
    
    for elm in tg_viborka:
        buf_predict = k_mean_model.predict([elm])
        buf_predict = int(buf_predict[0])
        pred.append(buf_predict)
    
    new_cluster = num_max(nn_predict_cluster.predict([pred]))
    
   
    prognos = nns_predicts_temp[new_cluster].predict([temp_to_prognos])
    
    return(prognos[0][0])

def accuracy(temperatures, predict, real):
    buf = ( abs(real - predict) ) / ( max(temperatures) - min(temperatures) ) * 100
    return(100 - buf)

def fallibility(temperatures, predict, real):
    buf = ( abs(real - predict) ) / ( max(temperatures) - min(temperatures) ) * 100
    return(buf)

def calculate_accuracy(temperatures, k_mean_model, nn_predict_cluster, nns_predicts_temp, size_of_vibor):
    x_to_list = []
    for i in range(len(temperatures) - 2*size_of_vibor):
        x_to_list.append(temperatures[i:i+size_of_vibor*2])
    real_to_list = []
    for i in range(size_of_vibor*2, len(temperatures)):
        real_to_list.append(temperatures[i])
    
    accurscy_list = []
    for i in range(len(real_to_list)):
        buffer_predict = predict_one_day(x_to_list[i], k_mean_model, model, NN, size_of_vibor)
        buffer_real = real_to_list[i]
        buffer_accuracy = accuracy(temperatures, buffer_predict, buffer_real)
        accurscy_list.append(buffer_accuracy)
        
    return(accurscy_list)

def srednee(arr):
    summ = 0
    for i in arr: summ += i
    return (summ / len(arr))

def calculate_fallibility(temperatures, k_mean_model, nn_predict_cluster, nns_predicts_temp, size_of_vibor):
    x_to_list = []
    for i in range(len(temperatures) - 2*size_of_vibor):
        x_to_list.append(temperatures[i:i+size_of_vibor*2])
    real_to_list = []
    for i in range(size_of_vibor*2, len(temperatures)):
        real_to_list.append(temperatures[i])
    
    fallibility_list = []
    for i in range(len(real_to_list)):
        buffer_predict = predict_one_day(x_to_list[i], k_mean_model, model, NN, size_of_vibor)
        buffer_real = real_to_list[i]
        buffer_fallibility = fallibility(temperatures, buffer_predict, buffer_real)
        fallibility_list.append(buffer_fallibility)
        
    return(fallibility_list)






#загрузка моделей

loaded_model = pickle.load(open(filename, 'rb'))        #к-средних
SUPPER = keras.models.load_model('model.h5')            #модель предсказания кластера
SUPER_NN = []                                           #модели предсказания значения
for i in range(clusters__):
    buf = keras.models.load_model('NN'+str(i)+'.h5')
    SUPER_NN.append(buf)





x = [-11.737499999999999,
 -4.3999999999999995,
 -1.975,
 -0.625,
 0.6125,
 0.6625,
 0.2875,
 -3.8875,
 -4.1,
 -2.5875000000000004]

x_real = -1.3375


def vis(x, loaded_model, SUPPER, SUPER_NN, size_of_vib, real_, pred_):
    x.append(real_)
    y = range(len(x))
    plt.plot(y,x)
    plt.scatter(len(y)-1, pred_, c = '#ff0000')
    plt.savefig('saved_figure.png' ,dpi = 65)

#vis(x, loaded_model, SUPPER, SUPER_NN, size_of_vib, x_real, tt)
    


#создание виджета окна

    

from tkinter import *
from PIL import Image, ImageTk

class App:

    def clicked(self):
        x = []

        x.append(float(self.txt1enter.get()))
        x.append(float(self.txt2enter.get()))
        x.append(float(self.txt3enter.get()))
        x.append(float(self.txt4enter.get()))
        x.append(float(self.txt5enter.get()))
        x.append(float(self.txt6enter.get()))
        x.append(float(self.txt7enter.get()))
        x.append(float(self.txt8enter.get()))
        x.append(float(self.txt9enter.get()))
        x.append(float(self.txt10enter.get()))

        x_real_ = float(self.preden.get())

        print(x)

        tt = predict_one_day(x, loaded_model, SUPPER, SUPER_NN, size_of_vib)
        vis(x, loaded_model, SUPPER, SUPER_NN, size_of_vib, x_real_, tt) #x_real

        self.image = Image.open("saved_figure.png")
        self.photo = ImageTk.PhotoImage(self.image)
        self.c_image = self.canvas.create_image(0, 0, anchor='nw', image=self.photo)
        self.canvas.grid(row=2, rowspan = 3, column=2, columnspan = 5)


        self.out1.configure(text=str(tt))

    def __init__(self):

        self.window = Tk()
        self.window.title("Предсказание погоды")


        self.lbl = Label(self.window, text="Введите данные")  
        self.lbl.grid(column=4, columnspan=2, row=0)  

        self.txt1enter = StringVar()
        self.txt1 = Entry(self.window, width = 10, textvariable=self.txt1enter)
        self.txt1.grid(row = 1, column = 0)

        self.txt2enter = StringVar()
        self.txt2 = Entry(self.window, width = 10, textvariable=self.txt2enter)
        self.txt2.grid(row = 1, column = 1)

        self.txt3enter = StringVar()
        self.txt3 = Entry(self.window, width = 10, textvariable=self.txt3enter)
        self.txt3.grid(row = 1, column = 2)

        self.txt4enter = StringVar()
        self.txt4 = Entry(self.window, width = 10, textvariable=self.txt4enter)
        self.txt4.grid(row = 1, column = 3)

        self.txt5enter = StringVar()
        self.txt5 = Entry(self.window, width = 10, textvariable=self.txt5enter)
        self.txt5.grid(row = 1, column = 4)

        self.txt6enter = StringVar()
        self.txt6 = Entry(self.window, width = 10, textvariable=self.txt6enter)
        self.txt6.grid(row = 1, column = 5)

        self.txt7enter = StringVar()
        self.txt7 = Entry(self.window, width = 10, textvariable=self.txt7enter)
        self.txt7.grid(row = 1, column = 6)

        self.txt8enter = StringVar()
        self.txt8 = Entry(self.window, width = 10, textvariable=self.txt8enter)
        self.txt8.grid(row = 1, column = 7)

        self.txt9enter = StringVar()
        self.txt9 = Entry(self.window, width = 10, textvariable=self.txt9enter)
        self.txt9.grid(row = 1, column = 8)

        self.txt10enter = StringVar()
        self.txt10 = Entry(self.window, width = 10, textvariable=self.txt10enter)
        self.txt10.grid(row = 1, column = 9)

        self.canvas = Canvas(self.window, height=300, width=400)
        self.canvas.grid(row=2, rowspan = 3, column=2, columnspan = 5)


        self.lbl1 = Label(self.window, text="Предсказанное занчение")  
        self.lbl1.grid(column=0, columnspan=3, row=6)  

        self.lbl2 = Label(self.window, text="Реальное занчение")  
        self.lbl2.grid(column=6, columnspan=3, row=6) 

        self.btn = Button(self.window, text="Запуск", command=self.clicked)  
        self.btn.grid(column=4, columnspan = 2, row=6)  

        self.out1 = Label(self.window, text="")  
        self.out1.grid(column=0, columnspan=2, row=7)   

        self.preden = StringVar()
        self.predtxt = Entry(self.window, width = 10, textvariable=self.preden)
        self.predtxt.grid(row = 7, column = 6, columnspan = 3)

        self.window.mainloop()


#цикл окна
app = App()


