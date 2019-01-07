#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 11:19:37 2019

@author: Alex Alves
"""
# Do keras
import matplotlib.pyplot as pl
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D

# Separando em conjunto de treinamento e teste
(entrada,desejado),(entrada_teste,desejado_teste) = mnist.load_data()

# Mostrando uma amostra colorida
#pl.imshow(entrada[2])

# Mostrando uma amostra em escala de cinza
#com seu respectivo valor no title
pl.imshow(entrada[2], cmap='gray')
pl.title('Valor '+str(desejado[2]))

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

