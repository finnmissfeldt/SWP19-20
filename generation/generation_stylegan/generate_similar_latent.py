
import facegeneration as fg
gan = fg.init()

import latent_finder_neural_network as nn
import sys
import PIL.Image as PIL
import numpy as np
import os


trainingFolder = "./training_data/"
resultFolder = "./results/"

def generateImage():
    global gan
    for i in range(1, len(sys.argv)):
        file = sys.argv[i]
        current_file = os.path.join(trainingFolder, file)
        img = PIL.open(current_file)
        img = img.convert('RGB')
        imgarray = np.array(img.resize((16, 16)))
        latent = nn.generate(imgarray)
        img_data = fg.generate(latent, gan)
        img = PIL.fromarray(img_data, 'RGB')
        img.save(resultFolder + "single_img_" + str(i) + ".png")

generateImage()

