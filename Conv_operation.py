"""
Author: Charlotte Hendrickx
Date: 11 May 2021
Implementation of a set of convolutions and pooling operations enabling to visualize the transformations applied to an image

Source: inspired by: https://mourafiq.com/2016/08/10/playing-with-convolutions-in-tensorflow.html
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import numpy


class Transformations:
    def __init__(self, image):
        self.padding = 'SAME'
        self.stride = (2,2)
        # Kernel
        a = np.zeros([3, 3, 3, 3])
        a[1, 1, :, :] = 5
        a[0, 1, :, :] = -1
        a[1, 0, :, :] = -1
        a[2, 1, :, :] = -1
        a[1, 2, :, :] = -1
        self.kernel = tf.constant(a, dtype=tf.float32)
        self.img = image

    def multilayer(self, n_layers):
        """Main function for performing the transformations.
        Input: number of layers of convolution + pooling you want to perform"""
        for i in range(1, n_layers+1):
            print("Layer nr: ", i)
            # For the first layer, take the input image
            if i == 1:
                # Convolution
                self.convolution(i, self.img)
                # Use the output of the convolution as input of the pooling layer
                img = Image.open(str("conv" + str(i) + ".png"))
                # Pooling
                self.pool(img.convert('RGB'), i)
            else:
                # Use the output of the pooling as input of the convolution layer
                img = Image.open(str("conv_pool" + str(i-1) + ".png"))
                # Convolution
                self.convolution(i, img.convert("RGB"))
                # Use the output of the convolution as input of the pooling layer
                img = Image.open(str("conv"+ str(i) + ".png"))
                # Pooling
                self.pool(img.convert('RGB'), i)



    def img_reshape(self, img):
        """Reshape the image to have a leading one dimension
        INPUT: image (pillow object)
        OUTPUT: reshaped image (np array)"""
        # reshape image to have a leading 1 dimension
        img = numpy.asarray(img, dtype='float32') / 256.
        img_shape = img.shape
        img_reshaped = img.reshape(1, img_shape[0], img_shape[1], 3)
        return img_reshaped

    def convolution(self, layer, img):
        """Convolution operation
        INPUT:  layer: int giving the number of convolutions+pooling operations done previously
                img: pillow object of the image to convolve
        OUTPUT: saves the convolved image to a file 'conv + layer + .png' """

        print("reshape conv", layer)
        # Reshape the pillow image to a numpy array
        img_reshaped = self.img_reshape(img)
        # Transform the kernel to a usable format to pass through the "conv2d" function
        w = tf.compat.v1.get_variable('w', initializer=tf.compat.v1.to_float(self.kernel))
        # Convolution operation
        conv = tf.nn.conv2d(input=img_reshaped, filters=w, strides=self.stride, padding=self.padding)
        # Saving the transformed, grayscale image
        plt.imsave(str("conv"+ str(layer) + ".png"), np.array(conv[0, :, :, 0], dtype='float32'), cmap=plt.get_cmap("gray"))


    def pool(self, img, layer):
        """Maxooling operation preceded by applying a sigmoid activation function
        INPUT:  img: pillow object of the image to pool
                layer: int giving the number of convolutions+pooling operations done previously
        OUTPUT: saves the pooled image to a file 'conv_pool + layer + .png' """

        print("reshape pool", layer)
        # Reshape the image
        img = self.img_reshape(img)
        # Apply the sigmoid activation function
        sig = tf.sigmoid(img)
        # Maxpooling
        max_pool = tf.nn.max_pool(sig, ksize=[1, 3, 3, 1], strides=self.stride, padding=self.padding)
        # Save the pooled, grayscale image
        plt.imsave(str("conv_pool" + str(layer) + ".png"), np.array(max_pool[0, :, :, 0], dtype='float32'), cmap=plt.get_cmap("gray"))





# Loading the original image
img = Image.open("ultimate_dog.jpg")

t = Transformations(img)
t.multilayer(3)
