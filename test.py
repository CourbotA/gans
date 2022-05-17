import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from keras.models import Model,Sequential,load_model
from keras.layers import * 

generator = load_model("./generator")
batch_size=10
z_dim=100
noise = np.random.normal(0, 1, (batch_size, z_dim))
labels = np.array([0,1,2,3,4,5,6,7,8,9]).reshape(-1, 1)

results = generator.predict([noise, labels])

print(results.shape)
plt.figure()
plt.subplot(2,5,1), plt.imshow( results[0],cmap='gray' )
plt.subplot(2,5,2), plt.imshow( results[1],cmap='gray' )
plt.subplot(2,5,3), plt.imshow( results[2],cmap='gray' )
plt.subplot(2,5,4), plt.imshow( results[3],cmap='gray' )
plt.subplot(2,5,5), plt.imshow( results[4],cmap='gray' )
plt.subplot(2,5,6), plt.imshow( results[5],cmap='gray' )
plt.subplot(2,5,7), plt.imshow( results[6],cmap='gray' )
plt.subplot(2,5,8), plt.imshow( results[7],cmap='gray' )
plt.subplot(2,5,9), plt.imshow( results[8],cmap='gray' )
plt.subplot(2,5,10), plt.imshow( results[9],cmap='gray' )
    
plt.show()