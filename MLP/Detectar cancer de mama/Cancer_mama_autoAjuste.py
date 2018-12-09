#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 09:23:17 2018

@author: Alex Alves

Programa para descobrir os melhores parametros
para treinar a rede neural
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
# Para selecionar os melhores parametros
from sklearn.model_selection import GridSearchCV 

entrada = pa.read_csv('entradas-breast.csv')
esperado = pa.read_csv('saidas-breast.csv')

# Passa como paramaetro para testar
def Criar_Rede(neuronios,func_activation,func_erro,inicializador,otimizador):
    # Criando a rede neural
    detectar_cancer = Sequential()
    #Adicionando camada de entrada 
    detectar_cancer.add(Dense(units=neuronios,activation=func_activation,kernel_initializer=inicializador,input_dim=30))
    
    # Para amenizar o overfiting faz o dropout(zerar algumas entradas)
    detectar_cancer.add(Dropout(0.2))
    
    
    #Adicionando uma camada oculta
    detectar_cancer.add(Dense(units=neuronios,activation=func_activation,kernel_initializer=inicializador))
    
    # Dropout para camada oculta
    detectar_cancer.add(Dropout(0.2))
    
    # Adicionando camada de saida
    detectar_cancer.add(Dense(units=1,activation='sigmoid'))
    
    
    
    detectar_cancer.compile(otimizador,loss=func_erro,metrics=['binary_accuracy'])

    return detectar_cancer

detectar_cancer =  KerasClassifier(build_fn=Criar_Rede,epochs=100,batch_size=10)
parametros = {'batch_size':[10,30],
              'epochs': [100,60],
              'neuronios': [8,16],
              'func_activation': ['relu','tanh'],
              'func_erro': ['hinge','binary_crossentropy'],
              'inicializador': ['normal','random_uniform'],
              'otimizador': ['adam','sgd'] } 

auto_ajuste=GridSearchCV(estimator=detectar_cancer,
                         param_grid=parametros,
                         scoring='accuracy',cv=5)


auto_ajuste=auto_ajuste.fit(entrada,esperado)
melhor_precisao=auto_ajuste.best_score_
melhores_parametros=auto_ajuste.best_params_



