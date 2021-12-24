
from PIL import Image
import numpy as np
from skimage import data
import matplotlib.pyplot as plt
from scipy import misc,ndimage
import PIL
import random
from PIL import ImageEnhance
image = data.camera()
type(image)#Image est un array numpy
"""numpy.ndarray"""


def add_salt_and_pepper(image, amount):

    output = np.copy(np.array(image))

    # add salt
    nb_salt = np.ceil(amount * output.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(nb_salt)) for i in output.shape]
    output[coords] = 1

    # add pepper
    nb_pepper = np.ceil(amount* output.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(nb_pepper)) for i in output.shape]
    output[coords] = 0

    return Image.fromarray(output)   


for i in range(2,20):
     
     im1 = Image.open(f'DatasetImage - Pierre/CNI{i}.png')
     ImageEnhance.Contrast(im1).enhance(1.8).save(f'DatasetImage - Pierre/CNI{i}-contrast.png')
     im1.convert('L').save(f'DatasetImage - Pierre/CNI{i}-yellow.png')
     im1.convert('1').save(f'DatasetImage - Pierre/CNI{i}-points.png')
     im1.convert('P').save(f'DatasetImage - Pierre/CNI{i}-lilnoisy.png')
     
     for k in range(1,11):
         im1.rotate(36*k).save(f'DatasetImage - Pierre/CNI{i}rotate@{k*36}.png')
         add_salt_and_pepper(im1,k/25 ).save(f'DatasetImage - Pierre/CNI{i}-noised@{k/25}.png')



# enh = ImageEnhance.Contrast(im1)
# enh.enhance(1.8).show("30% more contrast")
