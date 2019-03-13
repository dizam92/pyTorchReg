import os
from PIL import Image
import matplotlib.pyplot as plt

def display_logo():
    logo_path = './img/MILA.png'
    if not os.path.isfile(logo_path):
        logo_path = '/rap/colosse-users/GLO-4030/labs/img/MILA.png'
    img = Image.open(logo_path)
    plt.imshow(img)
    plt.axis('off')