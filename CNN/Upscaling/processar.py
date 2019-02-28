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

def Abrir_imagem(local,img):
    imagem = cv2.imread(local+"/"+img)
    return imagem

def Borrar(img):
    blur = cv2.blur(img,(7,7))
    return blur

def Noise(img):
    linha,coluna,c=img.shape
    gaussiano = np.round(np.random.rand(linha, coluna, 3) * 255).astype(np.uint8)
    ruido_gaussiano = cv2.addWeighted(img,0.5,gaussiano, 0.5, 0)
    #cv2.imshow("Original / Noise", np.hstack([img, gaussian_noise]))
    #cv2.waitKey(0)
    return ruido_gaussiano

def Resize(img):
    h, w,x = img.shape
    center = (w // 2, h // 2)
    re_img = cv2.resize(img,center ,fx=0.5,fy=0.5,interpolation = cv2.INTER_LINEAR)
    #re_img = cv2.resize(img,(w,h),interpolation = cv2.INTER_LINEAR)
    return re_img


def Salvar_resize(local,nome,img):
    cv2.imwrite(local+"/"+nome,img)
