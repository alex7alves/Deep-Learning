#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 15:46:54 2018

@author: Alex Alves

programa para treinar a rede com validação cruzada
usando o Dropout para amenizar o overfiting
"""


import pandas as pa

# Importação para poder  dividir os dados entre treinamento da rede e testes de validação
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from sklearn.metrics import confusion_matrix, accuracy_score
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score


entrada = pa.read_csv('entradas-breast.csv')
esperado = pa.read_csv('saidas-breast.csv')

def Criar_Rede():
    # Criando a rede neural
    detectar_cancer = Sequential()
    #Adicionando camada de entrada 
    detectar_cancer.add(Dense(units=16,activation='relu',kernel_initializer='random_uniform',input_dim=30))
    
    # Para amenizar o overfiting faz o dropout(zerar algumas entradas)
    detectar_cancer.add(Dropout(0.2))
    
    
    #Adicionando uma camada oculta
    detectar_cancer.add(Dense(units=16,activation='relu',kernel_initializer='random_uniform'))
    
    # Dropout para camada oculta
    detectar_cancer.add(Dropout(0.2))
    
    # Adicionando camada de saida
    detectar_cancer.add(Dense(units=1,activation='sigmoid'))
    
    # Compilar a rede
    #compile(descida_gradiente,função do erro- MSE, precisão da rede)
    
    # clipvalue -> delimita os valores dos pesos entre 0.5 e -0.5 
    # lr = tamanho do passo, decay-> redução do passo
    otimizar = keras.optimizers.Adam(lr=0.001,decay=0.0001)
    
    # Nesse caso o clipvalue prejudicou
    #otimizar = keras.optimizers.Adam(lr=0.004,decay=0.0001,clipvalue=0.5)
    
    detectar_cancer.compile(otimizar,loss='binary_crossentropy',metrics=['binary_accuracy'])

    return detectar_cancer


detectar_cancer =  KerasClassifier(build_fn=Criar_Rede,epochs=100,batch_size=10)
# cv = validação cruzada =10 vai dividir 10 vezes em treinamento e teste 
saida = cross_val_score(estimator=detectar_cancer,X=entrada,y=esperado,cv=10,scoring='accuracy')

acerto_medio=saida.mean()
desvio_padrao=saida.std()