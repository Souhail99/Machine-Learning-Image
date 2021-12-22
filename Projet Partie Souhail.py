import glob
import time
import numpy as np
from math import *
import csv
import pandas as pd
import matplotlib.pyplot as plt
import pytesseract
import shutil
import os
import random
try:
 from PIL import Image
except ImportError:
 import PIL
import glob
import cv2

from sklearn.model_selection import train_test_split

#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Dropout, Conv2DTranspose, Input
#from tensorflow.keras.callbacks import EarlyStopping




#On récupère dans une liste toute notre base de donées d'images
file="/home/guignole/Documents/Machine Learning Project/Machine-Learning-Image/DatasetImage - Souhail"
ext="/"
enfant="copie"
Listedesimages = [f for f in os.listdir(file) if f.endswith(".png")]
#print(Listedesimages," ",len(Listedesimages))
Listedesimages.sort()

#Maintenant nous devons ajouter à chaque élément son path pour python puisse le trouver
ListedesimagesavecExtension=[]
for i in Listedesimages:
    ListedesimagesavecExtension.append(file+ext+i)
#print(ListedesimagesavecExtension," fff ",len(ListedesimagesavecExtension))
img = Image.open(ListedesimagesavecExtension[0])

#On affiche la vrai image
img.show() 

filecsv="/home/guignole/Documents/Machine Learning Project/Machine-Learning-Image/Dataset/"

"""Ici on fait notre classe qui s'appelle individu, individu correspond à un fichier image et donc une carte d'identité dans la base de données."""

def LireCSV():
    Patronymes=pd.read_csv(filecsv+"patronymes.csv",sep=',')
    Prenom=pd.read_csv(filecsv+"prenom.csv",sep=',')
    PrenomAvecSexe=pd.read_csv(filecsv+"PrenomsEtSexe.csv",usecols=[2],sep=',')
    colonne=["01_prenom","02_genre"]
    #le genre est soit m (masculin) ou soit f (feminin)
    return Patronymes,Prenom,PrenomAvecSexe[colonne]

def Gray(img):
  image=Image.open(img)
  imagegray=image.convert('L')
  couleur='gray'
  l=img.split('.')
  imagegray.save(l[0]+couleur+l[1])
  lien=l[0]+couleur+l[1]
  return lien

def Rotation(img):
  image=Image.open(img)
  image180=image.rotate(180)
  rotation='rotation'
  l=img.split('.')
  image180.save(l[0]+rotation+l[1])
  lien=l[0]+rotation+l[1]
  return lien

class Individu:
  #Soit l'individu reste le meme soit il est modifié de manière aléatoire
    def __init__(self, a=None):
        pos=random.randint(0,len(ListedesimagesavecExtension)-1)
        self.a=ListedesimagesavecExtension[pos]
        change=random.randint(1,5)
        if change==1 or change==2:
          if change==1:
            self.a=Gray(self.a)
          if change==2:
            self.a=Rotation(self.a) 
    #On retourne le nom de l'image          
    def __str__(self):
        return str(f"L'image qui constitue cet individu est :{self.a}")
    #On affiche l'image
    def display(self):
        image=Image.open(self.a)
        return image.show()

    def levenshteinPrenom(prenom:str):
       taille_prenom1=len(prenom)+1
       beta=[]
       best=[]
       #On prend les 5 meillerus prenoms
       k=5
       Prenom=LireCSV()[1]
       for i in range(len(Prenom["prenom"])):
            ecart=0
            prenom2=Prenom["prenom"][i]
            taille_prenom2=len(prenom2)+1
            levenshtein_matrice=np.zeros((taille_prenom1,taille_prenom2))
            for x in range(taille_prenom1):
                levenshtein_matrice[x,0]=x
            for y in range(taille_prenom2):
                levenshtein_matrice[0,y]=y
            for x in range(1,taille_prenom1):
                for y in range(1,taille_prenom2):
                    if prenom[x-1]==prenom2[y-1]:
                        levenshtein_matrice[x,y]=min(
                    levenshtein_matrice[x-1, y] + 1,
                    levenshtein_matrice[x-1, y-1],
                    levenshtein_matrice[x, y-1] + 1)
                    
                    else:
                        levenshtein_matrice[x,y]=min(
                    levenshtein_matrice[x-1, y] + 1,
                    levenshtein_matrice[x-1, y-1] + 1,
                    levenshtein_matrice[x, y-1] + 1)
            ecart=levenshtein_matrice[prenom - 1, prenom2 - 1]
            beta.append([ecart,prenom,prenom2])
       beta = sorted(beta, key = lambda x : x[0])
       #best=[beta[x][-1] for x in range(k)]
       for x in range(k):
           best.append(beta[x][-1])
       return best

       
    
    def levenshteinNom(Nom:str):
       taille_Nom1=len(Nom)+1
       beta=[]
       best=[]
       #On prend les 5 meillerus prenoms
       k=5
       Patronymes=LireCSV()[0]
       for i in range(len(Patronymes["patronyme"])):
            ecart=0
            nom2=Patronymes["patronyme"][i]
            taille_nom2=len(nom2)+1
            levenshtein_matrice=np.zeros((taille_Nom1,taille_nom2))
            for x in range(taille_Nom1):
                levenshtein_matrice[x,0]=x
            for y in range(taille_nom2):
                levenshtein_matrice[0,y]=y
            for x in range(1,taille_Nom1):
                for y in range(1,taille_nom2):
                    if Nom[x-1]==nom2[y-1]:
                        levenshtein_matrice[x,y]=min(
                    levenshtein_matrice[x-1, y] + 1,
                    levenshtein_matrice[x-1, y-1],
                    levenshtein_matrice[x, y-1] + 1)
                    
                    else:
                        levenshtein_matrice[x,y]=min(
                    levenshtein_matrice[x-1, y] + 1,
                    levenshtein_matrice[x-1, y-1] + 1,
                    levenshtein_matrice[x, y-1] + 1)
            ecart=levenshtein_matrice[Nom - 1, nom2 - 1]
            beta.append([ecart,Nom,nom2])
       beta = sorted(beta, key = lambda x : x[0])
       for x in range(k):
           best.append(beta[x][-1])
       return best

    def PrenomAndNom(self,Prenom,Nom,sexe):
        #utilisation des distances de leven
        PrenomAvecSexe=LireCSV()[2]

        bestprenom=self.levenshteinPrenom(Prenom)
        bestNom=self.levenshteinNom(Nom)
        for j in bestprenom:
            for i in range(len(PrenomAvecSexe["01_prenom"])):
                if PrenomAvecSexe["01_prenom"][i]==j and PrenomAvecSexe["02_genre"][i]==sexe:
                    return j,bestNom[0]
        # for i in range(len(PrenomAvecSexe["01_prenom"])):
        #     for j in bestprenom:
        #         if PrenomAvecSexe["01_prenom"][i]==j and PrenomAvecSexe["02_genre"][i]==sexe:
        #             return j,bestNom[0]
        return bestprenom[0],bestNom[0]
    def ToString(self,Prenom,Nom,sexe):
        prenom,nom=self.PrenomAndNom(Prenom,Nom,sexe)
        return str(f"Le prenom cet individu est : {nom} et son prenom est : {prenom}")
    
#Faut aussi faire une fonction distance de leven avec juste tout les mots pour ville,prefectuere adresse etc
def levenshteinMot(Mot:str):
       taille_Mot1=len(Mot)+1
       beta=[]
       best=[]
       #On prend les 5 meillerus mots
       k=5
       
       #On liste le fichier texte dictionnaire mot français
       Mot=open(filecsv+"mot francais.txt", "r")
       Listedesmots=[]
       for f in Mot.readlines():
           Listedesmots.append(f.strip())

       for i in range(len(Listedesmots)):
            ecart=0
            Mot2=Listedesmots[i]
            taille_Mot2=len(Mot2)+1
            levenshtein_matrice=np.zeros((taille_Mot1,taille_Mot2))
            for x in range(taille_Mot1):
                levenshtein_matrice[x,0]=x
            for y in range(taille_Mot2):
                levenshtein_matrice[0,y]=y
            for x in range(1,taille_Mot1):
                for y in range(1,taille_Mot2):
                    if Mot[x-1]==Mot2[y-1]:
                        levenshtein_matrice[x,y]=min(
                    levenshtein_matrice[x-1, y] + 1,
                    levenshtein_matrice[x-1, y-1],
                    levenshtein_matrice[x, y-1] + 1)
                    
                    else:
                        levenshtein_matrice[x,y]=min(
                    levenshtein_matrice[x-1, y] + 1,
                    levenshtein_matrice[x-1, y-1] + 1,
                    levenshtein_matrice[x, y-1] + 1)
            ecart=levenshtein_matrice[Mot - 1, Mot2 - 1]
            beta.append([ecart,Mot,Mot2])
       beta = sorted(beta, key = lambda x : x[0])
       for x in range(k):
           best.append(beta[x][-1])
       return best[0]



#supprimer les caractères speciaux, espace et ajout espace entre avant nom et prenom
def SupCaracSpe(Mot:str):
    new_string = ''.join([l for l in Mot if l.isalnum() or l == ' '])
    new_string2=new_string.replace('  ',' ')
    new_string3=new_string2.replace('Nom',' Nom')
    new_string3=new_string3.replace('Pr',' Pr')
    return new_string3



"""Ici on réalise notre modèle :"""

#Partie Traitement d'image
#On reprend la variable img (sans qu'elle est soit modifié au préalable nécessaiire à comment notre modèle sera construit)
import easyocr

reader=easyocr.Reader(['fr'],gpu=False)
result=reader.readtext(img)
print(result)