import imp
from PIL import Image
import numpy as np 
import cv2 

img = Image.open('o_11.png')
if img.mode != 'L':
	img = img.convert('L')
	img = np.array(img)
	cv2.imwrite("o_11.png",img)