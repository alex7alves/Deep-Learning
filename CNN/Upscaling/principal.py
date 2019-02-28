 # -*- coding: utf-8 -*-

"""
Autor : Alex Alves
"""

import processar as p


def Capturar_imagens(local):
    treinamento=p.Retornar_imagens(local+"treinamento")
    teste=p.Retornar_imagens(local+"teste")
    return [treinamento,teste]


# Função para fazer resize em uma lista de imagens
def Formatar_imagens(path_inicio,path_destino,path_dados,lista):
    rand_array=p.Desordenar(lista)
    i=0
    for x in rand_array:
        imagem=p.Abrir_imagem(path_inicio,x)
        re_img=p.Resize(imagem)
        p.Salvar_resize(path_destino,x,re_img)
        resultado=Alterar_imagem(i,re_img)
        p.Salvar_resize(path_dados,x,resultado)
        i=i+1



def Alterar_imagem(z,img):
    if z==0:
        imagem = p.Borrar(img)
    elif z==1:
        imagem = p.Noise(img)
    else:
        image = p.Borrar(img)
        imagem = p.Noise(image)
    return imagem



# Carregar imanges
processar_treinamento,processar_teste = Capturar_imagens("Imagens/")
# Resize Imagens
Formatar_imagens("Imagens/treinamento","Resize/treinamento","Banco/treinamento",processar_treinamento)
Formatar_imagens("Imagens/teste","Resize/teste","Banco/teste",processar_teste)
print("Terminou ")
