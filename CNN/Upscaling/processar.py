 # -*- coding: utf-8 -*-

"""
Autor : Alex Alves
"""
import os
import cv2
import numpy as np
import random

# Função para capturar nomes de arquivos
def Retornar_imagens(caminho):
    for _, _, arquivo in os.walk(caminho):
        arq=arquivo
    return arquivo

# Função para deseordenar sequência de um lista
def Desordenar(lista):
    randon_array=set(lista)
    return randon_array

def Abrir_imagem(img):
    imagem = cv2.imread(img)
    return imagem

def Borrar(img):
    #imagem=Abrir_imagem(img)
    blur = cv2.blur(img,(7,7))
    return blur

def Noise(imga):
    img = cv2.imread('Imagens/ScreenShot027.jpg')
    print(img)
    linha,coluna,c=img.shape
    #gaussian = np.int_(np.random.random((linha, coluna, 3))*10)
    gaussian = np.round(np.random.rand(linha, coluna, 3) * 255).astype(np.uint8)
    gaussian_noise = cv2.addWeighted(img,0.5,gaussian, 0.5, 0)
    cv2.imshow("Original / Noise", np.hstack([img, gaussian_noise]))
    cv2.waitKey(0)

def Reduzir(img):
    #h, w,x = img.shape
    #center = (w // 2, h // 2)
    #re_img = cv2.resize(img,center ,fx=0.5,fy=0.5,interpolation = cv2.INTER_LINEAR)
    re_img = cv2.resize(img,(640,640))
    cv2.imshow("Show by CV2",re_img)
    cv2.waitKey(0)
    return re_img

imagens=Retornar_imagens('Imagens/')
#print(imagens)
array_set=Desordenar(imagens)
#print(array_set)

image=Abrir_imagem('Imagens/ScreenShot027.jpg')
#Borrar(image)
Reduzir(image)
#Noise(image)
