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
img = ListedesimagesavecExtension[5]

#On affiche la vrai image


import easyocr

reader=easyocr.Reader(['en'],gpu=False)
result=reader.readtext(img)
print(result)