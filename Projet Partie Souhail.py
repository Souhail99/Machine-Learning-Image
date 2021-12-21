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

filecsv="/home/guignole/Documents/Machine Learning Project/Machine-Learning-Image/Dataset"

"""Ici on fait notre classe qui s'appelle individu, individu correspond à un fichier image et donc une carte d'identité dans la base de données."""

def LireCSV():
    Patronymes=pd.read_csv(filecsv+"patronymes.csv",sep=',')
    Prenom=pd.read_csv(filecsv+"prenom.csv",sep=',')
    PrenomAvecSexe=pd.read_csv(filecsv+"Prenoms.csv",usecols=[2],sep=',')
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
        for i in range(len(PrenomAvecSexe["01_prenom"])):
            for j in bestprenom:
                if PrenomAvecSexe["01_prenom"][i]==j and PrenomAvecSexe["02_genre"][i]==sexe:
                    return j,bestNom[0]
        return bestprenom[0],bestNom[0]
    def ToString(self,Prenom,Nom,sexe):
        prenom,nom=self.PrenomAndNom(Prenom,Nom,sexe)
        return str(f"Le prenom cet individu est : {nom} et son prenom est : {prenom}")
    

"""On fait aussi quelques tests ici :"""

im = cv2.imread("/content/CNI3.png", cv2.IMREAD_COLOR)

im = cv2.bitwise_not(im)
plt.imshow(im)
plt.show()

image_path_in_colab="/content/CNI3.png"
extract = pytesseract.image_to_string(Image.open(image_path_in_colab))
print(extract)



def ocr_core(img):
  text=pytesseract.image_to_string(img)
  return text
#img=Image.open('/content/CNI2.png')
img=cv2.imread('/content/CNI3.png')

def get_grayscale(img):
  return cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

def remove_noise(img):
  return cv2.medianBlur(img,5)

def thresholding(img):
  return cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

img=get_grayscale(img)
img=thresholding(img)
img=remove_noise(img)

print("new version :",ocr_core(img),'alpha')

"""Ici on réalise notre modèle :"""

