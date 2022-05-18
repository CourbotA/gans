from dis import dis
from itertools import cycle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras.layers as layers
from tensorflow.keras.optimizers import Adam
import keras

from os import listdir
from numpy import asarray
from numpy import vstack
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from numpy import savez_compressed

### CLASS ###
class ResBlock(keras.Model):
    def __init__(self, filters):
        super().__init__()

        self.conv0 = keras.Sequential([
            layers.Conv2D(filters, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.ReLU()
        ])

        self.conv1 = keras.Sequential([
            layers.Conv2D(filters, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.ReLU()
        ])

    def call(self, input):
        result = self.conv0(input)
        result = self.conv1(result)

        input = input + result

        return input

    def model(self, input_shape):
        x = keras.Input(input_shape)

        return keras.models.Model(x, self.call(x))

class Generator(keras.Model):
    def __init__(self, n_resnet=9, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.conv0 = keras.Sequential([
            layers.Conv2D(64, (7, 7), padding='same'),
            layers.BatchNormalization(),
            layers.ReLU()
        ])

        self.conv1 = keras.Sequential([
            layers.Conv2D(128, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.ReLU()
        ])

        self.conv2 = keras.Sequential([
            layers.Conv2D(256, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.ReLU()
        ])

        self.resnet = keras.Sequential()

        for _ in range(n_resnet):
            self.resnet.add(ResBlock(256))

        self.convTrans0 = keras.Sequential([
            layers.Conv2DTranspose(128, (3,3), strides=(2,2), padding='same'),
            layers.BatchNormalization(),
            layers.ReLU()
        ])

        self.convTrans1 = keras.Sequential([
            layers.Conv2DTranspose(64, (3,3), strides=(2,2), padding='same'),
            layers.BatchNormalization(),
            layers.ReLU()
        ])

        self.last = keras.Sequential([
            layers.Conv2D(3, (7, 7), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('tanh')
        ])
        

    def call(self, inputs, training=None, mask=None):
        inputs = self.conv0(inputs)
        inputs = self.conv1(inputs)
        inputs = self.conv2(inputs)
        inputs = self.resnet(inputs)
        inputs = self.convTrans0(inputs)
        inputs = self.convTrans1(inputs)
        inputs = self.last(inputs)
        
        return inputs

    def model(self, input_shape):
        x = keras.Input(input_shape)
        return keras.models.Model(x, self.call(x))


class Discriminator(keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.conv64 = keras.Sequential([
            layers.Conv2D(64, (4,4), strides=(2,2), padding='same'),
            layers.LeakyReLU(alpha=0.2)
        ])

        self.conv128 = keras.Sequential([
            layers.Conv2D(128, (4,4), strides=(2,2), padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2)
        ])

        self.conv256 = keras.Sequential([
            layers.Conv2D(256, (4,4), strides=(2,2), padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2)
        ])

        self.conv512_1 = keras.Sequential([
            layers.Conv2D(512, (4,4), strides=(2,2), padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2)
        ])

        self.conv512_2 = keras.Sequential([
            layers.Conv2D(512, (4,4), padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2)
        ])

        self.last = keras.Sequential([
            layers.Conv2D(1, (4,4), padding='same')
        ])

    def call(self, inputs, training=None, mask=None):
        inputs = self.conv64(inputs)
        inputs = self.conv128(inputs)
        inputs = self.conv256(inputs)
        inputs = self.conv512_1(inputs)
        inputs = self.conv512_2(inputs)
        inputs = self.last(inputs)

        return inputs

    def model(self, input_shape):
        x = keras.Input(input_shape)
        return keras.models.Model(x, self.call(x))

class CycleGAN(keras.Model):
    def __init__(self, gen_1, gen_2, disc, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.gen_1 = gen_1

        self.gen_2 = gen_2

        self.disc = disc

    def call(self, inputs, training=None, mask=None):
        input_gen, input_id = inputs
        #discriminator
        gen1_out = self.gen_1(input_gen)
        output_d = self.disc(gen1_out)

        #identity
        output_id = self.gen_1(input_id)

        #forward cycle
        output_f = self.gen_2(gen1_out)

        #backward cycle
        gen2_out = self.gen_2(input_id)
        output_b = self.gen_1(gen2_out)

        return [output_d, output_id, output_f, output_b]

    def model(self, input_shape):
        gen = keras.Input(input_shape)
        id = keras.Input(input_shape)
        return keras.models.Model([gen, id], self.call([gen, id]))

### CODE ###

# load and prepare training images
def load_real_samples(filename):
	# load the dataset
	data = np.load(filename)
	# unpack arrays
	X1, X2 = data['arr_0'], data['arr_1']
	# scale from [0,255] to [-1,1]
	X1 = (X1 - 127.5) / 127.5
	X2 = (X2 - 127.5) / 127.5
	return [X1, X2]

# select a batch of random samples, returns images and target
def generate_real_samples(dataset, n_samples, patch_shape):
	# choose random instances
	ix = np.randint(0, dataset.shape[0], n_samples)
	# retrieve selected images
	X = dataset[ix]
	# generate 'real' class labels (1)
	y = np.ones((n_samples, patch_shape, patch_shape, 1))
	return X, y

# generate a batch of images, returns images and targets
def generate_fake_samples(g_model, dataset, patch_shape):
	# generate fake instance
	X = g_model.predict(dataset)
	# create 'fake' class labels (0)
	y = np.zeros((len(X), patch_shape, patch_shape, 1))
	return X, y

# save the generator models to file
def save_models(step, g_model_AtoB, g_model_BtoA):
	# save the first generator model
	filename1 = 'g_model_AtoB_%06d.h5' % (step+1)
	g_model_AtoB.save(filename1)
	# save the second generator model
	filename2 = 'g_model_BtoA_%06d.h5' % (step+1)
	g_model_BtoA.save(filename2)
	print('>Saved: %s and %s' % (filename1, filename2))

# generate samples and save as a plot and save the model
def summarize_performance(step, g_model, trainX, name, n_samples=5):
	# select a sample of input images
	X_in, _ = generate_real_samples(trainX, n_samples, 0)
	# generate translated images
	X_out, _ = generate_fake_samples(g_model, X_in, 0)
	# scale all pixels from [-1,1] to [0,1]
	X_in = (X_in + 1) / 2.0
	X_out = (X_out + 1) / 2.0
	# plot real images
	for i in range(n_samples):
		plt.subplot(2, n_samples, 1 + i)
		plt.axis('off')
		plt.imshow(X_in[i])
	# plot translated image
	for i in range(n_samples):
		plt.subplot(2, n_samples, 1 + n_samples + i)
		plt.axis('off')
		plt.imshow(X_out[i])
	# save plot to file
	filename1 = '%s_generated_plot_%06d.png' % (name, (step+1))
	plt.savefig(filename1)
	plt.close()

# update image pool for fake images
def update_image_pool(pool, images, max_size=50):
	selected = list()
	for image in images:
		if len(pool) < max_size:
			# stock the pool
			pool.append(image)
			selected.append(image)
		elif np.random() < 0.5:
			# use image, but don't add it to the pool
			selected.append(image)
		else:
			# replace an existing image and use replaced image
			ix = np.randint(0, len(pool))
			selected.append(pool[ix])
			pool[ix] = image
	return asarray(selected)

# train cyclegan models
def train(d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, c_model_AtoB, c_model_BtoA, dataset):
	# define properties of the training run
	n_epochs, n_batch, = 100, 1
	# determine the output square shape of the discriminator
	n_patch = d_model_A.output_shape[1]
	# unpack dataset
	trainA, trainB = dataset
	# prepare image pool for fakes
	poolA, poolB = list(), list()
	# calculate the number of batches per training epoch
	bat_per_epo = int(len(trainA) / n_batch)
	# calculate the number of training iterations
	n_steps = bat_per_epo * n_epochs
	# manually enumerate epochs
	for i in range(n_steps):
		# select a batch of real samples
		X_realA, y_realA = generate_real_samples(trainA, n_batch, n_patch)
		X_realB, y_realB = generate_real_samples(trainB, n_batch, n_patch)
		# generate a batch of fake samples
		X_fakeA, y_fakeA = generate_fake_samples(g_model_BtoA, X_realB, n_patch)
		X_fakeB, y_fakeB = generate_fake_samples(g_model_AtoB, X_realA, n_patch)
		# update fakes from pool
		X_fakeA = update_image_pool(poolA, X_fakeA)
		X_fakeB = update_image_pool(poolB, X_fakeB)
		# update generator B->A via adversarial and cycle loss
		g_loss2, _, _, _, _  = c_model_BtoA.train_on_batch([X_realB, X_realA], [y_realA, X_realA, X_realB, X_realA])
		# update discriminator for A -> [real/fake]
		dA_loss1 = d_model_A.train_on_batch(X_realA, y_realA)
		dA_loss2 = d_model_A.train_on_batch(X_fakeA, y_fakeA)
		# update generator A->B via adversarial and cycle loss
		g_loss1, _, _, _, _ = c_model_AtoB.train_on_batch([X_realA, X_realB], [y_realB, X_realB, X_realA, X_realB])
		# update discriminator for B -> [real/fake]
		dB_loss1 = d_model_B.train_on_batch(X_realB, y_realB)
		dB_loss2 = d_model_B.train_on_batch(X_fakeB, y_fakeB)
		# summarize performance
		print('>%d, dA[%.3f,%.3f] dB[%.3f,%.3f] g[%.3f,%.3f]' % (i+1, dA_loss1,dA_loss2, dB_loss1,dB_loss2, g_loss1,g_loss2))
		# evaluate the model performance every so often
		if (i+1) % (bat_per_epo * 1) == 0:
			# plot A->B translation
			summarize_performance(i, g_model_AtoB, trainA, 'AtoB')
			# plot B->A translation
			summarize_performance(i, g_model_BtoA, trainB, 'BtoA')
		if (i+1) % (bat_per_epo * 5) == 0:
			# save the models
			save_models(i, g_model_AtoB, g_model_BtoA)


# load image data
dataset = load_real_samples('horse2zebra_256.npz')
print('Loaded', dataset[0].shape, dataset[1].shape)
# define input shape based on the loaded dataset
image_shape = dataset[0].shape[1:]
# generator: A -> B
g_model_AtoB = Generator()
g_model_AtoB = g_model_AtoB.model(image_shape)
# generator: B -> A
g_model_BtoA = Generator()
g_model_BtoA = g_model_BtoA.model(image_shape)
# discriminator: A -> [real/fake]
d_model_A = Discriminator()
d_model_A = d_model_A.model(image_shape)
d_model_A.compile(loss='mse', optimizer=Adam(learning_rate=0.0002, beta_1=0.5), loss_weights=[0.5])
# discriminator: B -> [real/fake]
d_model_B = Discriminator()
d_model_B = d_model_B.model(image_shape)
d_model_B.compile(loss='mse', optimizer=Adam(learning_rate=0.0002, beta_1=0.5), loss_weights=[0.5])
# composite: A -> B -> [real/fake, A]
c_model_AtoB = CycleGAN(g_model_AtoB, g_model_BtoA, d_model_B)
c_model_AtoB = c_model_AtoB.model(image_shape)
c_model_AtoB.compile(loss=['mse', 'mae', 'mae', 'mae'], loss_weights=[1, 5, 10, 10], optimizer=Adam(learning_rate=0.0002, beta_1=0.5))
# composite: B -> A -> [real/fake, B]
c_model_BtoA = CycleGAN(g_model_BtoA, g_model_AtoB, d_model_A)
c_model_BtoA = c_model_BtoA.model(image_shape)
c_model_BtoA.compile(loss=['mse', 'mae', 'mae', 'mae'], loss_weights=[1, 5, 10, 10], optimizer=Adam(learning_rate=0.0002, beta_1=0.5))
# train models
train(d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, c_model_AtoB, c_model_BtoA, dataset)