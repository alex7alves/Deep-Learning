#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 20:19:41 2019

@author: Alex Alves

Rede neural convolucional com validação cruzada
para treinar digitos escritos a mão
"""

import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from sklearn.model_selection import StratifiedKFold
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import np_utils

semente = 5
np.random.seed(semente)

# Separando em conjunto de treinamento e teste
(entrada,desejado),(entrada_teste,desejado_teste) = mnist.load_data()

# Mudando o formato 
x_treinamento = entrada.reshape(entrada.shape[0],28,28,1)
# Passando formato de unit8 para float32 (para normalizar)
x_treinamento = x_treinamento.astype('float32')
# Fazendo a normalização 
x_treinamento=x_treinamento/255
# Transformando o desejado em categoria (1 -> 1 0 0 0 0 0 0 0 0 0)
d_treinamento = np_utils.to_categorical(desejado,10) 

# prepara base para separar para validação cruzada
separar_base = StratifiedKFold(n_splits = 5, shuffle = True, random_state = semente)
saida = []

b = np.zeros(shape = (d_treinamento.shape[0], 1))

for i_treinamento, i_teste in separar_base.split(x_treinamento, 
                                            np.zeros(shape = (d_treinamento.shape[0], 1))):

    digitos = Sequential()
    digitos.add(Conv2D(32,(3,3), input_shape=(28,28,1), activation='relu'))
    digitos.add(MaxPooling2D(pool_size=(2,2)))
    digitos.add(Flatten())

    digitos.add(Dense(units=128,activation='relu'))

    digitos.add(Dense(units=10,activation='softmax'))
    digitos.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    digitos.fit(x_treinamento[i_treinamento], d_treinamento[i_treinamento],
                      batch_size = 128, epochs = 5)
    precisao = digitos.evaluate(x_treinamento[i_teste], d_treinamento[i_teste])
    saida.append(precisao[1])

media = sum(saida) / len(saida)
