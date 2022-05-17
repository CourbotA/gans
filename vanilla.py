import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from keras.models import Model,Sequential
from keras.layers import * 
l=tf.keras.layers


def build_discriminator(input_shape=(28, 28,), verbose=True):

    """
    Utility method to build a MLP discriminator
    Parameters:
    input_shape:
    type:tuple. Shape of input image for classification.
    Default shape is (28,28)->MNIST
    verbose:
    type:boolean. Print model summary if set to true.
    Default is True
    Returns:
    tensorflow.keras.model object
    """
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(l.Flatten())
    model.add(l.Dense(512))
    model.add(l.LeakyReLU(alpha=0.2))
    model.add(l.Dense(256))
    model.add(l.LeakyReLU(alpha=0.2))
    model.add(l.Dense(1, activation='sigmoid'))
    if verbose:
         model.summary()

    return model

def build_generator(z_dim=100, output_shape=(28, 28), verbose=True):
    """
    Utility method to build a MLP generator
    Parameters:
    z_dim:
    type:int(positive). Size of input noise vector to be
    used as model input.
    Default value is 100
    output_shape: type:tuple. Shape of output image .
    Default shape is (28,28)->MNIST
    verbose:
    type:boolean. Print model summary if set to true.
    Default is True
    Returns:
    tensorflow.keras.model object
    """
    model = Sequential()
    model.add(Input(shape=(z_dim,)))
    model.add(l.Dense(256, input_dim=z_dim))
    model.add(l.LeakyReLU(alpha=0.2))
    model.add(l.BatchNormalization(momentum=0.8))
    model.add(l.Dense(512))
    model.add(l.LeakyReLU(alpha=0.2))
    model.add(l.BatchNormalization(momentum=0.8))
    model.add(l.Dense(1024))
    model.add(l.LeakyReLU(alpha=0.2))
    model.add(l.BatchNormalization(momentum=0.8))
    model.add(l.Dense(np.prod(output_shape), activation='tanh'))
    model.add(l.Reshape(output_shape))
    if verbose:
        model.summary()
    return model

def build_dc_generator(z_dim=100, verbose=True):
    model = Sequential()
    model.add(Input(shape=(z_dim,)))
    model.add(Dense(128 * 7 * 7, activation="relu", input_dim=z_dim))
    model.add(Reshape((7, 7, 128)))
    model.add(UpSampling2D())
    model.add(Conv2D(128, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
    model.add(UpSampling2D())
    model.add(Conv2D(64, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
    model.add(Conv2D(1, kernel_size=3, padding="same"))
    model.add(Activation("tanh"))
    if verbose:
        model.summary()
    return model

def build_conditional_generator(z_dim=100, output_shape=(28, 28),num_classes=10, verbose=True):
    """
    Utility method to build a MLP generator
    Parameters:
        z_dim:
        type:int(positive). Size of input noise vector to be used as model input. 
        Default value is 100
    output_shape: type:tuple. Shape of output image .
    Default shape is (28,28)->MNIST
    num_classes: type:int. Number of unique class labels.
    Default is 10->MNIST digits
    verbose:
    type:boolean. Print model summary if set to true.
    Default is True
    Returns:
    tensorflow.keras.model object
    """
    noise = Input(shape=(z_dim,))
    label = Input(shape=(1,), dtype='int32')
    label_embedding = Flatten()(Embedding(num_classes, z_dim)(label))
    model_input = multiply([noise, label_embedding])
    mlp = Dense(256, input_dim=z_dim)(model_input)
    mlp = LeakyReLU(alpha=0.2)(mlp)
    mlp = BatchNormalization(momentum=0.8)(mlp)
    mlp = Dense(512)(mlp)
    mlp = LeakyReLU(alpha=0.2)(mlp)
    mlp = Dense(1024)(mlp)
    mlp = LeakyReLU(alpha=0.2)(mlp)
    mlp = BatchNormalization(momentum=0.8)(mlp)
    mlp = Dense(np.prod(output_shape), activation='tanh')(mlp)
    mlp = Reshape(output_shape)(mlp)
    model = Model([noise, label], mlp)
    if verbose:
        model.summary()
    return model

def build_conditional_discriminator(input_shape=(28, 28,),num_classes=10, verbose=True):
    """
    Utility method to build a conditional MLP discriminator
    Parameters:
    input_shape:
    type:tuple. Shape of input image for classification.
    Default shape is (28,28)->MNIST
    num_classes: type:int. Number of unique class labels.
    Default is 10->MNIST digits
    verbose:
    type:boolean. Print model summary if set to true.
    Default is True
    Returns:
    tensorflow.keras.model object
    """
    img = Input(shape=input_shape)
    flat_img = Flatten()(img)
    label = Input(shape=(1,), dtype='int32')
    label_embedding = Flatten()(Embedding(num_classes,
    np.prod(input_shape))(label))
    model_input = multiply([flat_img, label_embedding])

    mlp = Dense(512, input_dim=np.prod(input_shape))(model_input)
    mlp = LeakyReLU(alpha=0.2)(mlp)
    mlp = Dense(512)(mlp)
    mlp = LeakyReLU(alpha=0.2)(mlp)
    mlp = Dropout(0.4)(mlp)
    mlp = Dense(512)(mlp)
    mlp = LeakyReLU(alpha=0.2)(mlp)
    mlp = Dropout(0.4)(mlp)
    mlp = Dense(1, activation='sigmoid')(mlp)
    model = Model([img, label], mlp)
    if verbose:
        model.summary()
    return model

def train(generator=None,discriminator=None,gan_model=None,epochs=1000, batch_size=128, sample_interval=50,z_dim=100):
    # Load MNIST train samples
    (X_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()
    # Rescale -1 to 1
    X_train = X_train / 127.5 - 1
    X_train = np.expand_dims(X_train, axis=3)
    y_train = y_train.reshape(-1, 1)
    # Prepare GAN output labels
    real_y = np.ones((batch_size, 1))
    fake_y = np.zeros((batch_size, 1))
    for epoch in range(epochs):
        # train disriminator
        # pick random real samples from X_train
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        real_imgs, labels = X_train[idx], y_train[idx]
        # pick random noise samples (z) from a normal distribution
        noise = np.random.normal(0, 1, (batch_size, z_dim))
        # use generator model to generate output samples
        fake_imgs = generator.predict([noise, labels])
        # calculate discriminator loss on real samples
        disc_loss_real = discriminator.train_on_batch([real_imgs,
        labels], real_y)
        # calculate discriminator loss on fake samples
        disc_loss_fake = discriminator.train_on_batch([fake_imgs,
        labels], fake_y)
        # overall discriminator loss
        discriminator_loss = 0.5 * np.add(disc_loss_real, disc_loss_fake)
        # train generator
        # pick random noise samples (z) from a normal distribution
        noise = np.random.normal(0, 1, (batch_size, z_dim))
        # pick random labels for conditioning
        sampled_labels = np.random.randint(0, 10, batch_size).reshape(-1, 1)
        # use trained discriminator to improve generator
        gen_loss = gan_model.train_on_batch([noise, sampled_labels],
        real_y)
        # training updates
        print ("%d [Discriminator loss: %f, acc.: %.2f%%] [Generatorloss: %f]" % (epoch, discriminator_loss[0],100*discriminator_loss[1], gen_loss))
        #If at save interval => save generated image samples
        if epoch % sample_interval == 0:
            sample_images(epoch,generator)

train(generator=build_conditional_discriminator(),discriminator=build_discriminator())

if (generator):
    generator.save(".\generator")
    discriminator.save(".\discriminator")
