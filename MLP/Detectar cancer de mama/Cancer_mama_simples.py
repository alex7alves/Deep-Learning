#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 21:04:48 2018

@author: Alex Alves

Programa para determinar se um tumor de mama 
é benigno (saida 0) ou maligno (saida 1)

"""
import pandas as pa

# Importação para poder  dividir os dados entre treinamento da rede e testes de validação
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense

from sklearn.metrics import confusion_matrix, accuracy_score


entrada = pa.read_csv('entradas-breast.csv')
esperado = pa.read_csv('saidas-breast.csv')


# Treinamento com 75% e validação com 25%
entrada_treinar, entrada_teste, esperado_treinar,esperado_teste =train_test_split(entrada,esperado,test_size=0.25)

# Criando a rede neural
detectar_cancer = Sequential()
#Adicionando camada de entrada 
detectar_cancer.add(Dense(units=16,activation='relu',kernel_initializer='random_uniform',input_dim=30))

#Adicionando uma camada oculta
detectar_cancer.add(Dense(units=16,activation='relu',kernel_initializer='random_uniform'))

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

#detectar_cancer.compile(optimizer='adam',loss='binary_crossentropy',metrics=['binary_accuracy'])
#  Fazer o treinamento da rede - erro calculado para 10 amostras 
#depois atualiza os pesos -descida do gradiente estocasticos de 10 em 10 amostras
detectar_cancer.fit(entrada_treinar,esperado_treinar,batch_size=10,epochs=100)


# Realizando teste de validação
# retorna probabilidade de acerto
validar = detectar_cancer.predict(entrada_teste)

# convertendo para true ou false (1 ou 0) para comparar
# se for maior que 0.5 é true, caso contrário é false
validar = (validar > 0.5)

# compara os 2 vetores e calcula a porcentagem de acerto
# da rede usando o conjunto de treinamento 
precisao = accuracy_score(esperado_teste,validar)

# Matriz de acertos da rede
acertos = confusion_matrix(esperado_teste,validar)

# Outra maneira de resultado
# retorna o erro e a precisão
resultado = detectar_cancer.evaluate(entrada_teste, esperado_teste)

