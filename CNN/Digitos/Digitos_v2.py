#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 11:19:37 2019

@author: Alex Alves

Rede neural convolucional para reconhecer digitos.



"""
# Do keras
import matplotlib.pyplot as pl
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization


# Separando em conjunto de treinamento e teste
(entrada,desejado),(entrada_teste,desejado_teste) = mnist.load_data()


# Mudando o formato 
#reshape(numero de registro,altura,largura,canal -> 1= tom de cinza )
x_treinamento = entrada.reshape(entrada.shape[0],28,28,1)
x_teste = entrada_teste.reshape(entrada_teste.shape[0],28,28,1)
# Passando formato de unit8 para float32 (para normalizar)
x_treinamento = x_treinamento.astype('float32')
x_teste = x_teste.astype('float32')

# Fazendo a normalização 
x_treinamento=x_treinamento/255
x_teste=x_teste/255

# Transformando o desejado em categoria (1 -> 1 0 0 0 0 0 0 0 0 0)
d_treinamento = np_utils.to_categorical(desejado,10) 
d_teste = np_utils.to_categorical(desejado_teste,10)

digitos = Sequential()
# Camada de convolução (eatapa 1) com 32 detectores de caracteristicas 
# conv2D(numero de filtros, tamanho da janela)
digitos.add(Conv2D(32,(3,3), input_shape=(28,28,1), activation='relu'))

# Normalização para camada de convolução
digitos.add(BatchNormalization())

# Pooling (etapa 2)
digitos.add(MaxPooling2D(pool_size=(2,2)))

# Adicionando camada de convolução  
digitos.add(Conv2D(32,(3,3), activation='relu'))
digitos.add(BatchNormalization())
digitos.add(MaxPooling2D(pool_size=(2,2)))

# Flattening (etapa 3)
digitos.add(Flatten())
# Etapa 4
# Camada escondida
digitos.add(Dense(units=128,activation='relu'))
digitos.add(Dropout(0.2))
# Addicionando camada oculta
digitos.add(Dense(units=128,activation='relu'))
digitos.add(Dropout(0.2))
# Camada de saida
digitos.add(Dense(units=10,activation='softmax'))
digitos.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
digitos.fit(x_treinamento,d_treinamento,batch_size=128,epochs=5,
            validation_data=(x_teste,d_teste))

