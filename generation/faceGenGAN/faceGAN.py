import os
os.environ['CUDA_VISIBLE_DEVICES'] = "2"

import sys
from PIL import Image
from numpy import asarray
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Embedding
from keras.layers import Reshape
from keras.layers import Concatenate
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Dropout
from keras.models import Model
from keras.optimizers import Adam
import numpy as np
import os
import random
from numpy import expand_dims
from numpy import zeros
from numpy import ones
from numpy.random import randn
from numpy.random import randint
from keras.layers import Flatten

#import json



FILEF = "2ganFem.h5"
FILEM = "2ganMale.h5"
IMG_SIZE = 28
latent_dim = 100

#generate points in latent space as input for the generator
def generate_latent_point(latent_dim, n_samples):
  x_input = randn(latent_dim * n_samples)
  x_input = x_input.reshape(n_samples, latent_dim)
  return x_input

  
def define_generator(latent_dim):
	model = Sequential()

	n_nodes = 128 * 7 * 7
	model.add(Dense(n_nodes, input_dim=latent_dim))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Reshape((7, 7, 128)))

	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))

	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))

	model.add(Conv2D(1, (7,7), activation='tanh', padding='same'))
	return model


#create the generator
generator  = define_generator(latent_dim)

#read JSON file INACTIVE
#with open('./file.json') as f: #replace file with real filename
#	data = json.load(f)

#gender = json.dumps(data)

fileName = "txtFile"

num = sys.argv[1]

fileName += num

fileName += ".txt"

print(fileName)

gender = ""

try:
  f = open(fileName, "r")
  gender = f.read(1)
except (OSError, IOError) as e:
  print("Error: Ein solcher Dateiname existiert nicht.")
  sys.exit(1)


if gender == "f":
  print("load female weights")
  generator.load_weights(FILEF)
elif gender == "m":
  print("load male weights")
  generator.load_weights(FILEM)
else:
  print("Error")
  sys.exit(1)
	
   
latent_point = generate_latent_point(100,100)
  
X = generator.predict([latent_point])
#scale from [-1,1] => [0,1]
X = (X + 1) / 2.0
#print(X[0])
#img = X.reshape((128, 28 ,28))
#print(img)
imgarr = np.ndarray(shape=(IMG_SIZE, IMG_SIZE), dtype=np.uint8)
# Convert array
for x in range(0, len(X[0])):
  for y in range(0, len(X[0][x])):
    imgarr[x][y] = (np.uint8) (255 * X[0][x][y][0])

img = Image.fromarray(imgarr, 'L')
img = img.convert("L")
img.save('newFace.png')
  
