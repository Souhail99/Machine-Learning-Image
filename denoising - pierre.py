import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image  
# Reading image from folder where it is stored
# img = Image.open(f'DatasetImage - Pierre/CNI2.png')
# img.show()
# img =np.asarray(img)

# img2 = cv2.cvtColor(np.asarray(img), cv2.COLOR_GRAY2BGR)
# # denoising of image saving it into dst image

#Image.fromarray(dst).show()
# 24

import glob
cv_img = []
j=1
for img in glob.glob('DatasetImage - Pierre/*.png'):
    print(j)
    n= cv2.imread(img)
    dst = cv2.fastNlMeansDenoisingColored(n, None, 10, 10, 7, 15)
    dst=Image.fromarray(dst)
    dst.save(f'filterredPierro/{j}.png')
    j+=1
