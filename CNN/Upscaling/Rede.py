"""

Autor : Alex Alves

Programa para treinar rede CNN para upscaling
"""
import pandas as pa

# Importação para poder  dividir os dados entre treinamento da rede e testes de validação
from sklearn.model_selection import train_test_split
import keras


from sklearn.metrics import confusion_matrix, accuracy_score

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.layers.normalization import BatchNormalization
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

def Normalizar(x):
    for i in range(len(x)):
        x[i] = x[i].astype('float32')
        x[i] = x[i]/255
    return x

# retorna em (w,h,c) do opencv
entrada_treinamento = Carregar_imagens("Banco/treinamento")
desejado_treinamento = Carregar_imagens("Imagens/treinamento")
entrada_teste = Carregar_imagens("Banco/teste")
desejado_teste = Carregar_imagens("Imagens/teste")
#print(desejado_treinamento[0].shape)
#print(len(desejado_treinamento))

entrada_treinamento = Normalizar(entrada_treinamento)
desejado_treinamento = Normalizar(desejado_treinamento)
entrada_teste = Normalizar(entrada_teste)
desejado_teste = Normalizar(desejado_teste)
#print(entrada_teste[0])





w,h,c=entrada_treinamento[0].shape
#print(w,h,c)
upscaling = Sequential()
# Opencv abre imagem de forma invertida (w,h,c)
upscaling.add(Conv2D(16, (3,3), input_shape = (w, h, c), activation = 'relu'))
upscaling.add(BatchNormalization())

upscaling.add(Flatten())

upscaling.add(Dense(units = 128, activation = 'relu'))
#upscaling.add(Dropout(0.2))
upscaling.add(Dense(units = 128, activation = 'relu'))
#upscaling.add(Dropout(0.2))
upscaling.add(Dense(units = 1, activation = 'sigmoid'))

upscaling.compile(optimizer = 'adam', loss = 'binary_crossentropy',
                      metrics = ['accuracy'])







#gerador_treinamento = ImageDataGenerator(rescale = 1./255)
#base_treinamento = gerador_treinamento.flow_from_directory('Banco/')
