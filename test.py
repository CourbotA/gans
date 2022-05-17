import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from keras.models import Model,Sequential,load_model
from keras.layers import * 

generator = load_model(".\generator")
discriminator = load_model(".\discriminator")
batch_size=128
z_dim=100
noise = np.random.normal(0, 1, (batch_size, z_dim))
for j in range(1,32):
    plt.figure()
    plt.imshow( generator.predict(noise)[j],cmap='gray' )
    
plt.show()