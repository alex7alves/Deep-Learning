#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 09:08:47 2019

@author: alex
"""


import pandas as pa

# Importação para poder  dividir os dados entre treinamento da rede e testes de validação
from sklearn.model_selection import train_test_split
import keras


from sklearn.metrics import confusion_matrix, accuracy_score

from keras.models import Model,Sequential
from keras.layers import InputLayer, Input,Conv2DTranspose, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.preprocessing import image


import processar as p

def Carregar_imagens(caminho):
    lista_imagens=[]
    arquivos =p.Retornar_imagens(caminho)
    for x in arquivos:
        img =p.Abrir_imagem(caminho,x)
        lista_imagens.append(img)
    return lista_imagens

def NormalizarLista(x):
    for i in range(len(x)):
        x[i] = x[i].astype('float32')
        x[i] = x[i]/255
    return x
def Normalizar(x): 
    x = x.astype('float32')
    x = x/255
    return x

def Converter(x):
    w,h,c=x[0].shape
    x = np.array(x, dtype=np.uint8)
    x = x.reshape(x.shape[0],w,h,3)
    return x


# retorna em (w,h,c) do opencv
entrada_treinamento = Carregar_imagens("Banco/treinamento")
desejado_treinamento = Carregar_imagens("Imagens/treinamento")
entrada_teste = Carregar_imagens("Banco/teste")
desejado_teste = Carregar_imagens("Imagens/teste")


entrada_treinamento=Converter(entrada_treinamento)
desejado_treinamento=Converter(desejado_treinamento)

entrada_teste=Converter(entrada_teste)
desejado_teste=Converter(desejado_teste)



entrada_treinamento = Normalizar(entrada_treinamento)
desejado_treinamento = Normalizar(desejado_treinamento)
entrada_teste = Normalizar(entrada_teste)
desejado_teste = Normalizar(desejado_teste)



autoencoder = Sequential()
# Encoder
autoencoder.add(Conv2D(filters = 16, kernel_size = (3,3), activation = 'relu', input_shape=(120,120,3),strides = (2,2),padding='same'))
autoencoder.add(Conv2D(filters = 8, kernel_size = (3,3), activation = 'relu',strides = (2,2),padding='same'))
autoencoder.add(Flatten())
autoencoder.summary()
autoencoder.add(Reshape((30,30,8)))
autoencoder.add(Conv2DTranspose(filters=8,kernel_size=(3,3),strides=(2,2),activation='relu',padding='same'))
autoencoder.add(Conv2DTranspose(filters=16,kernel_size=(3,3),strides=(2,2),activation='relu',padding='same'))
autoencoder.summary()
autoencoder.add(Conv2DTranspose(filters=3,kernel_size=(3,3),strides=(2,2),activation='sigmoid',padding='same'))

autoencoder.summary()


autoencoder.compile(optimizer = 'adam', loss = 'mean_squared_error',
                   metrics = ['accuracy'])

autoencoder.fit(entrada_treinamento,desejado_treinamento,
                epochs = 500,
                validation_data = (entrada_teste,desejado_teste))


'''

autoencoder = Sequential()
# Encoder
autoencoder.add(Conv2D(filters = 16, kernel_size = (5,5), activation = 'relu', input_shape=(84,127,3),strides = (2,2),padding='same'))
autoencoder.add(Conv2D(filters = 8, kernel_size = (5,5), activation = 'relu',strides = (2,2),padding='same'))
autoencoder.add(Flatten())
autoencoder.summary()
autoencoder.add(Reshape((21,32,8)))
autoencoder.add(Conv2DTranspose(filters=8,kernel_size=(3,3),strides=(2,2),activation='relu',padding='same'))
autoencoder.add(Conv2DTranspose(filters=16,kernel_size=(3,3),strides=(2,2),activation='relu',padding='same'))
autoencoder.summary()
autoencoder.add(Conv2DTranspose(filters=3,kernel_size=(3,3),strides=(2,2),activation='sigmoid',padding='same'))

autoencoder.summary()
'''