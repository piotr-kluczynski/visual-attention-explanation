# LIBRARIES IMPORT
import keras
import os
import pickle
import tensorflow as tf
import pandas as pd

from keras import callbacks
from keras.saving import save_model
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization, Reshape, GlobalAveragePooling2D, GlobalMaxPooling2D, Multiply, Concatenate, Input

# ATTENTION LAYERS DEFINITIONS
class Channel_Attention(layers.Layer):
    def __init__(self, channels, reduction, cbam_id):
        super().__init__(name=f"cbam{cbam_id}_channel")

        # Property for saving last attention map
        self.channel_attention_map = None

        self.avg_pool = GlobalAveragePooling2D(name=f"cbam{cbam_id}_channel_global_avg_pool")
        self.max_pool = GlobalMaxPooling2D(name=f"cbam{cbam_id}_channel_global_max_pool")

        self.dense1 = Dense(channels // reduction, activation='relu', name=f"cbam{cbam_id}_channel_dense1")
        self.dense2 = Dense(channels, activation=None, name=f"cbam{cbam_id}_channel_dense2")

        self.reshape = Reshape((1, 1, channels), name=f"cbam{cbam_id}_channel_reshape")
        self.multiply = Multiply(name=f"cbam{cbam_id}_channel_multiply")

    def build(self, input_shape):
        x_shape = input_shape
        
        self.avg_pool.build(input_shape)
        self.avg_pool.build(input_shape)
        x_shape = self.avg_pool.compute_output_shape(input_shape)

        self.dense1.build(x_shape)
        x_shape = self.dense1.compute_output_shape(x_shape)

        self.dense2.build(x_shape)
        x_shape = self.dense2.compute_output_shape(x_shape)
    
    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, input, save_attention=False):
        x = input

        avg_pool = self.avg_pool(x)
        avg_pool = self.dense1(avg_pool)
        avg_pool = self.dense2(avg_pool)
        
        max_pool = self.max_pool(x)
        max_pool = self.dense1(max_pool)
        max_pool = self.dense2(max_pool)

        x = avg_pool + max_pool
        x = tf.nn.sigmoid(x)
        x = self.reshape(x)

        if save_attention:
            self.channel_attention_map = x

        return self.multiply([input, x])


class Spatial_Attention(layers.Layer):
    def __init__(self, kernel_size, cbam_id):
        super().__init__(name=f"cbam{cbam_id}_spatial")

        # Property for saving last attention map
        self.spatial_attention_map = None

        self.conv = Conv2D(1, kernel_size, strides=1, padding='same', activation='sigmoid', name=f"cbam{cbam_id}_spatial_conv")
        
        self.concat = Concatenate(axis=-1, name=f"cbam{cbam_id}_spatial_concatenate")
        self.multiply = Multiply(name=f"cbam{cbam_id}_spatial_multiply")

        self.attention = None # Value of spatial attention weights after last call()

    def build(self, input_shape):
        x_shape = (input_shape[0], input_shape[1], input_shape[2], 2)
        self.conv.build(x_shape)


    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, input, save_attention=False):
        x = input

        avg_pool = tf.reduce_mean(x, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(x, axis=-1, keepdims=True)

        x = self.concat([avg_pool, max_pool])

        x = self.conv(x)

        if save_attention:
            self.spatial_attention_map = x
        
        return self.multiply([input, x])


class CBAM(layers.Layer):
    def __init__(self, channels, cbam_id, reduction=8, kernel_size=7):
        super().__init__(name=f"CBAM_{cbam_id}")

        self.channel = Channel_Attention(channels, reduction, cbam_id=cbam_id)
        self.spatial = Spatial_Attention(kernel_size, cbam_id=cbam_id)

    def build(self, input_shape):
        self.channel.build(input_shape)
        self.spatial.build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, input, save_attention=False):
        x = input

        x = self.channel(x, save_attention=save_attention)
        x = self.spatial(x, save_attention=save_attention)

        return x


class Squeeze_Excite(layers.Layer):
    def __init__(self, channels, se_id, ratio = 16):
        super().__init__(name=f"SE_{se_id}")

        # Property for saving last attention map
        self.attention_map = None
        
        self.pool = layers.GlobalAveragePooling2D(name=f"se{se_id}_global_avg_pool")                 
        self.dense1 = layers.Dense(channels // ratio, activation='relu', name=f"se{se_id}_dense1")
        self.dense2 = layers.Dense(channels, activation='sigmoid', name=f"se{se_id}_dense2")

    def compute_output_shape(self, input_shape):
        return input_shape

    def build(self, input_shape):
        x_shape = input_shape

        self.pool.build(x_shape)
        x_shape = self.pool.compute_output_shape(x_shape)

        self.dense1.build(x_shape)
        x_shape = self.dense1.compute_output_shape(x_shape)

        self.dense2.build(x_shape)
        x_shape = self.dense2.compute_output_shape(x_shape)

    def call(self, input, save_attention=False):
        x = self.pool(input)
        x = self.dense1(x)
        x = self.dense2(x)
        x = tf.expand_dims(tf.expand_dims(x, 1), 1)

        if save_attention:
            self.attention_map = x

        return input * x


# MODEL DEFINITIONS
@keras.saving.register_keras_serializable(package="Custom")
class Baseline_Model(keras.Model):
    def __init__(self, name="BaselineModel", **kwargs):
        super().__init__(**kwargs)
        self.name = name

        # VGG19
        # Block 1
        self.vgg19_block1_conv1 = Conv2D(64, (3, 3), activation="relu", padding="same", name="block1_conv1")
        self.vgg19_block1_conv2 = Conv2D(64, (3, 3), activation="relu", padding="same", name="block1_conv2")
        self.vgg19_block1_pool = MaxPooling2D((2, 2), strides=(2, 2), name="block1_pool")

        # Block 2
        self.vgg19_block2_conv1 = Conv2D(128, (3, 3), activation="relu", padding="same", name="block2_conv1")
        self.vgg19_block2_conv2 = Conv2D(128, (3, 3), activation="relu", padding="same", name="block2_conv2")
        self.vgg19_block2_pool = MaxPooling2D((2, 2), strides=(2, 2), name="block2_pool")

        # Block 3
        self.vgg19_block3_conv1 = Conv2D(256, (3, 3), activation="relu", padding="same", name="block3_conv1")
        self.vgg19_block3_conv2 = Conv2D(256, (3, 3), activation="relu", padding="same", name="block3_conv2")
        self.vgg19_block3_conv3 = Conv2D(256, (3, 3), activation="relu", padding="same", name="block3_conv3")
        self.vgg19_block3_conv4 = Conv2D(256, (3, 3), activation="relu", padding="same", name="block3_conv4")
        self.vgg19_block3_pool = MaxPooling2D((2, 2), strides=(2, 2), name="block3_pool")
        
        # Block 4
        self.vgg19_block4_conv1 = Conv2D(512, (3, 3), activation="relu", padding="same", name="block4_conv1")
        self.vgg19_block4_conv2 = Conv2D(512, (3, 3), activation="relu", padding="same", name="block4_conv2")
        self.vgg19_block4_conv3 = Conv2D(512, (3, 3), activation="relu", padding="same", name="block4_conv3")
        self.vgg19_block4_conv4 = Conv2D(512, (3, 3), activation="relu", padding="same", name="block4_conv4")
        self.vgg19_block4_pool = MaxPooling2D((2, 2), strides=(2, 2), name="block4_pool")

        # Block 5
        self.vgg19_block5_conv1 = Conv2D(512, (3, 3), activation="relu", padding="same", name="block5_conv1")
        self.vgg19_block5_conv2 = Conv2D(512, (3, 3), activation="relu", padding="same", name="block5_conv2")
        self.vgg19_block5_conv3 = Conv2D(512, (3, 3), activation="relu", padding="same", name="block5_conv3")
        self.vgg19_block5_conv4 = Conv2D(512, (3, 3), activation="relu", padding="same", name="block5_conv4")
        self.vgg19_block5_pool = MaxPooling2D((2, 2), strides=(2, 2), name="block5_pool")

        # Classifier layer
        self.class_flatten = Flatten(name="class_flatten")
        
        self.class_dense1 = Dense(512, activation = 'relu', name="class_dense1")
        self.class_dropout1 = Dropout(0.4, name="class_dropout1")
        self.class_batch1 = BatchNormalization(name="class_batch1")
        
        self.class_dense2 = Dense(512, activation = 'relu', name="class_dense2")
        self.class_dropout2 = Dropout(0.3, name="class_dropout2")
        self.class_batch2 = BatchNormalization(name="class_batch2")

        self.class_dense3 = Dense(26, activation = 'softmax', name="class_dense3")

    def call(self, inputs, training=False):
        # VGG19
        # Block 1
        x = self.vgg19_block1_conv1(inputs)
        x = self.vgg19_block1_conv2(x)
        x = self.vgg19_block1_pool(x)

        # Block 2
        x = self.vgg19_block2_conv1(x)
        x = self.vgg19_block2_conv2(x)
        x = self.vgg19_block2_pool(x)

        # Block 3
        x = self.vgg19_block3_conv1(x)
        x = self.vgg19_block3_conv2(x)
        x = self.vgg19_block3_conv3(x)
        x = self.vgg19_block3_conv4(x)
        x = self.vgg19_block3_pool(x)

        # Block 4
        x = self.vgg19_block4_conv1(x)
        x = self.vgg19_block4_conv2(x)
        x = self.vgg19_block4_conv3(x)
        x = self.vgg19_block4_conv4(x)
        x = self.vgg19_block4_pool(x)

        # Block 5
        x = self.vgg19_block5_conv1(x)
        x = self.vgg19_block5_conv2(x)
        x = self.vgg19_block5_conv3(x)
        x = self.vgg19_block5_conv4(x)
        x = self.vgg19_block5_pool(x)

        # Classifier layer
        x = self.class_flatten(x)

        x = self.class_dense1(x)
        x = self.class_dropout1(x, training=training)
        x = self.class_batch1(x, training=training)

        x = self.class_dense2(x)
        x = self.class_dropout2(x, training=training)
        x = self.class_batch2(x, training=training)

        x = self.class_dense3(x)

        return x

    def build(self, input_shape):
        # VGG19
        # Block 1
        self.vgg19_block1_conv1.build(input_shape)
        x_shape = self.vgg19_block1_conv1.compute_output_shape(input_shape)
        self.vgg19_block1_conv2.build(x_shape)
        x_shape = self.vgg19_block1_conv2.compute_output_shape(x_shape)
        self.vgg19_block1_pool.build(x_shape)
        x_shape = self.vgg19_block1_pool.compute_output_shape(x_shape)

        # Block 2
        self.vgg19_block2_conv1.build(x_shape)
        x_shape = self.vgg19_block2_conv1.compute_output_shape(x_shape)
        self.vgg19_block2_conv2.build(x_shape)
        x_shape = self.vgg19_block2_conv2.compute_output_shape(x_shape)
        self.vgg19_block2_pool.build(x_shape)
        x_shape = self.vgg19_block2_pool.compute_output_shape(x_shape)

        # Block 3
        self.vgg19_block3_conv1.build(x_shape)
        x_shape = self.vgg19_block3_conv1.compute_output_shape(x_shape)
        self.vgg19_block3_conv2.build(x_shape)
        x_shape = self.vgg19_block3_conv2.compute_output_shape(x_shape)
        self.vgg19_block3_conv3.build(x_shape)
        x_shape = self.vgg19_block3_conv3.compute_output_shape(x_shape)
        self.vgg19_block3_conv4.build(x_shape)
        x_shape = self.vgg19_block3_conv4.compute_output_shape(x_shape)
        self.vgg19_block3_pool.build(x_shape)
        x_shape = self.vgg19_block3_pool.compute_output_shape(x_shape)

        # Block 4
        self.vgg19_block4_conv1.build(x_shape)
        x_shape = self.vgg19_block4_conv1.compute_output_shape(x_shape)
        self.vgg19_block4_conv2.build(x_shape)
        x_shape = self.vgg19_block4_conv2.compute_output_shape(x_shape)
        self.vgg19_block4_conv3.build(x_shape)
        x_shape = self.vgg19_block4_conv3.compute_output_shape(x_shape)
        self.vgg19_block4_conv4.build(x_shape)
        x_shape = self.vgg19_block4_conv4.compute_output_shape(x_shape)
        self.vgg19_block4_pool.build(x_shape)
        x_shape = self.vgg19_block4_pool.compute_output_shape(x_shape)

        # Block 5
        self.vgg19_block5_conv1.build(x_shape)
        x_shape = self.vgg19_block5_conv1.compute_output_shape(x_shape)
        self.vgg19_block5_conv2.build(x_shape)
        x_shape = self.vgg19_block5_conv2.compute_output_shape(x_shape)
        self.vgg19_block5_conv3.build(x_shape)
        x_shape = self.vgg19_block5_conv3.compute_output_shape(x_shape)
        self.vgg19_block5_conv4.build(x_shape)
        x_shape = self.vgg19_block5_conv4.compute_output_shape(x_shape)
        self.vgg19_block5_pool.build(x_shape)
        x_shape = self.vgg19_block5_pool.compute_output_shape(x_shape)

        # Classifier layer
        self.class_flatten.build(x_shape)
        x_shape = self.class_flatten.compute_output_shape(x_shape)
        self.class_dense1.build(x_shape)
        x_shape = self.class_dense1.compute_output_shape(x_shape)
        self.class_dropout1.build(x_shape)
        x_shape = self.class_dropout1.compute_output_shape(x_shape)
        self.class_batch1.build(x_shape)
        x_shape = self.class_batch1.compute_output_shape(x_shape)
        
        self.class_dense2.build(x_shape)
        x_shape = self.class_dense2.compute_output_shape(x_shape)
        self.class_dropout2.build(x_shape)
        x_shape = self.class_dropout2.compute_output_shape(x_shape)
        self.class_batch2.build(x_shape)
        x_shape = self.class_batch2.compute_output_shape(x_shape)

        self.class_dense3.build(x_shape)


@keras.saving.register_keras_serializable(package="Custom")
class SE_Model(keras.Model):
    def __init__(self, name="SeModel", **kwargs):
        super().__init__(**kwargs)
        self.name = name
        
        # VGG19
        # Block 1
        self.vgg19_block1_conv1 = Conv2D(64, (3, 3), activation="relu", padding="same", name="block1_conv1")
        self.vgg19_block1_conv2 = Conv2D(64, (3, 3), activation="relu", padding="same", name="block1_conv2")
        self.vgg19_block1_attention = Squeeze_Excite(64, se_id='block1')
        self.vgg19_block1_pool = MaxPooling2D((2, 2), strides=(2, 2), name="block1_pool")

        # Block 2
        self.vgg19_block2_conv1 = Conv2D(128, (3, 3), activation="relu", padding="same", name="block2_conv1")
        self.vgg19_block2_conv2 = Conv2D(128, (3, 3), activation="relu", padding="same", name="block2_conv2")
        self.vgg19_block2_attention = Squeeze_Excite(128, se_id='block2')
        self.vgg19_block2_pool = MaxPooling2D((2, 2), strides=(2, 2), name="block2_pool")

        # Block 3
        self.vgg19_block3_conv1 = Conv2D(256, (3, 3), activation="relu", padding="same", name="block3_conv1")
        self.vgg19_block3_conv2 = Conv2D(256, (3, 3), activation="relu", padding="same", name="block3_conv2")
        self.vgg19_block3_conv3 = Conv2D(256, (3, 3), activation="relu", padding="same", name="block3_conv3")
        self.vgg19_block3_conv4 = Conv2D(256, (3, 3), activation="relu", padding="same", name="block3_conv4")
        self.vgg19_block3_attention = Squeeze_Excite(256, se_id='block3')
        self.vgg19_block3_pool = MaxPooling2D((2, 2), strides=(2, 2), name="block3_pool")
        
        # Block 4
        self.vgg19_block4_conv1 = Conv2D(512, (3, 3), activation="relu", padding="same", name="block4_conv1")
        self.vgg19_block4_conv2 = Conv2D(512, (3, 3), activation="relu", padding="same", name="block4_conv2")
        self.vgg19_block4_conv3 = Conv2D(512, (3, 3), activation="relu", padding="same", name="block4_conv3")
        self.vgg19_block4_conv4 = Conv2D(512, (3, 3), activation="relu", padding="same", name="block4_conv4")
        self.vgg19_block4_attention = Squeeze_Excite(512, se_id='block4')
        self.vgg19_block4_pool = MaxPooling2D((2, 2), strides=(2, 2), name="block4_pool")

        # Block 5
        self.vgg19_block5_conv1 = Conv2D(512, (3, 3), activation="relu", padding="same", name="block5_conv1")
        self.vgg19_block5_conv2 = Conv2D(512, (3, 3), activation="relu", padding="same", name="block5_conv2")
        self.vgg19_block5_conv3 = Conv2D(512, (3, 3), activation="relu", padding="same", name="block5_conv3")
        self.vgg19_block5_conv4 = Conv2D(512, (3, 3), activation="relu", padding="same", name="block5_conv4")
        self.vgg19_block5_attention = Squeeze_Excite(512, se_id='block5')
        self.vgg19_block5_pool = MaxPooling2D((2, 2), strides=(2, 2), name="block5_pool")

        # Classifier layer
        self.class_flatten = Flatten(name="class_flatten")
        
        self.class_dense1 = Dense(512, activation = 'relu', name="class_dense1")
        self.class_dropout1 = Dropout(0.4, name="class_dropout1")
        self.class_batch1 = BatchNormalization(name="class_batch1")
        
        self.class_dense2 = Dense(512, activation = 'relu', name="class_dense2")
        self.class_dropout2 = Dropout(0.3, name="class_dropout2")
        self.class_batch2 = BatchNormalization(name="class_batch2")

        self.class_dense3 = Dense(LABEL_CLASS, activation = 'softmax', name="class_dense3")

    def call(self, inputs, save_attention=False, training=False):
        # VGG19
        # Block 1
        x = self.vgg19_block1_conv1(inputs)
        x = self.vgg19_block1_conv2(x)
        x = self.vgg19_block1_attention(x, save_attention=save_attention)
        x = self.vgg19_block1_pool(x)

        # Block 2
        x = self.vgg19_block2_conv1(x)
        x = self.vgg19_block2_conv2(x)
        x = self.vgg19_block2_attention(x, save_attention=save_attention)
        x = self.vgg19_block2_pool(x)

        # Block 3
        x = self.vgg19_block3_conv1(x)
        x = self.vgg19_block3_conv2(x)
        x = self.vgg19_block3_conv3(x)
        x = self.vgg19_block3_conv4(x)
        x = self.vgg19_block3_attention(x, save_attention=save_attention)
        x = self.vgg19_block3_pool(x)

        # Block 4
        x = self.vgg19_block4_conv1(x)
        x = self.vgg19_block4_conv2(x)
        x = self.vgg19_block4_conv3(x)
        x = self.vgg19_block4_conv4(x)
        x = self.vgg19_block4_attention(x, save_attention=save_attention)
        x = self.vgg19_block4_pool(x)

        # Block 5
        x = self.vgg19_block5_conv1(x)
        x = self.vgg19_block5_conv2(x)
        x = self.vgg19_block5_conv3(x)
        x = self.vgg19_block5_conv4(x)
        x = self.vgg19_block5_attention(x, save_attention=save_attention)
        x = self.vgg19_block5_pool(x)

        # Classifier layer
        x = self.class_flatten(x)

        x = self.class_dense1(x)
        x = self.class_dropout1(x, training=training)
        x = self.class_batch1(x, training=training)

        x = self.class_dense2(x)
        x = self.class_dropout2(x, training=training)
        x = self.class_batch2(x, training=training)

        x = self.class_dense3(x)

        return x

    def build(self, input_shape):
        # VGG19
        # Block 1
        self.vgg19_block1_conv1.build(input_shape)
        x_shape = self.vgg19_block1_conv1.compute_output_shape(input_shape)
        self.vgg19_block1_conv2.build(x_shape)
        x_shape = self.vgg19_block1_conv2.compute_output_shape(x_shape)
        self.vgg19_block1_attention.build(x_shape)
        x_shape = self.vgg19_block1_attention.compute_output_shape(x_shape)
        self.vgg19_block1_pool.build(x_shape)
        x_shape = self.vgg19_block1_pool.compute_output_shape(x_shape)

        # Block 2
        self.vgg19_block2_conv1.build(x_shape)
        x_shape = self.vgg19_block2_conv1.compute_output_shape(x_shape)
        self.vgg19_block2_conv2.build(x_shape)
        x_shape = self.vgg19_block2_conv2.compute_output_shape(x_shape)
        self.vgg19_block2_attention.build(x_shape)
        x_shape = self.vgg19_block2_attention.compute_output_shape(x_shape)
        self.vgg19_block2_pool.build(x_shape)
        x_shape = self.vgg19_block2_pool.compute_output_shape(x_shape)

        # Block 3
        self.vgg19_block3_conv1.build(x_shape)
        x_shape = self.vgg19_block3_conv1.compute_output_shape(x_shape)
        self.vgg19_block3_conv2.build(x_shape)
        x_shape = self.vgg19_block3_conv2.compute_output_shape(x_shape)
        self.vgg19_block3_conv3.build(x_shape)
        x_shape = self.vgg19_block3_conv3.compute_output_shape(x_shape)
        self.vgg19_block3_conv4.build(x_shape)
        x_shape = self.vgg19_block3_conv4.compute_output_shape(x_shape)
        self.vgg19_block3_attention.build(x_shape)
        x_shape = self.vgg19_block3_attention.compute_output_shape(x_shape)
        self.vgg19_block3_pool.build(x_shape)
        x_shape = self.vgg19_block3_pool.compute_output_shape(x_shape)

        # Block 4
        self.vgg19_block4_conv1.build(x_shape)
        x_shape = self.vgg19_block4_conv1.compute_output_shape(x_shape)
        self.vgg19_block4_conv2.build(x_shape)
        x_shape = self.vgg19_block4_conv2.compute_output_shape(x_shape)
        self.vgg19_block4_conv3.build(x_shape)
        x_shape = self.vgg19_block4_conv3.compute_output_shape(x_shape)
        self.vgg19_block4_conv4.build(x_shape)
        x_shape = self.vgg19_block4_conv4.compute_output_shape(x_shape)
        self.vgg19_block4_attention.build(x_shape)
        x_shape = self.vgg19_block4_attention.compute_output_shape(x_shape)
        self.vgg19_block4_pool.build(x_shape)
        x_shape = self.vgg19_block4_pool.compute_output_shape(x_shape)

        # Block 5
        self.vgg19_block5_conv1.build(x_shape)
        x_shape = self.vgg19_block5_conv1.compute_output_shape(x_shape)
        self.vgg19_block5_conv2.build(x_shape)
        x_shape = self.vgg19_block5_conv2.compute_output_shape(x_shape)
        self.vgg19_block5_conv3.build(x_shape)
        x_shape = self.vgg19_block5_conv3.compute_output_shape(x_shape)
        self.vgg19_block5_conv4.build(x_shape)
        x_shape = self.vgg19_block5_conv4.compute_output_shape(x_shape)
        self.vgg19_block5_attention.build(x_shape)
        x_shape = self.vgg19_block5_attention.compute_output_shape(x_shape)
        self.vgg19_block5_pool.build(x_shape)
        x_shape = self.vgg19_block5_pool.compute_output_shape(x_shape)

        # Classifier layer
        self.class_flatten.build(x_shape)
        x_shape = self.class_flatten.compute_output_shape(x_shape)
        self.class_dense1.build(x_shape)
        x_shape = self.class_dense1.compute_output_shape(x_shape)
        self.class_dropout1.build(x_shape)
        x_shape = self.class_dropout1.compute_output_shape(x_shape)
        self.class_batch1.build(x_shape)
        x_shape = self.class_batch1.compute_output_shape(x_shape)
        
        self.class_dense2.build(x_shape)
        x_shape = self.class_dense2.compute_output_shape(x_shape)
        self.class_dropout2.build(x_shape)
        x_shape = self.class_dropout2.compute_output_shape(x_shape)
        self.class_batch2.build(x_shape)
        x_shape = self.class_batch2.compute_output_shape(x_shape)

        self.class_dense3.build(x_shape)


@keras.saving.register_keras_serializable(package="Custom")
class CBAM_Model(keras.Model):
    def __init__(self, name="CbamModel", **kwargs):
        super().__init__(**kwargs)
        self.name = name
        
        # VGG19
        # Block 1
        self.vgg19_block1_conv1 = Conv2D(64, (3, 3), activation="relu", padding="same", name="block1_conv1")
        self.vgg19_block1_conv2 = Conv2D(64, (3, 3), activation="relu", padding="same", name="block1_conv2")
        self.vgg19_block1_attention = CBAM(64, cbam_id='block1')
        self.vgg19_block1_pool = MaxPooling2D((2, 2), strides=(2, 2), name="block1_pool")

        # Block 2
        self.vgg19_block2_conv1 = Conv2D(128, (3, 3), activation="relu", padding="same", name="block2_conv1")
        self.vgg19_block2_conv2 = Conv2D(128, (3, 3), activation="relu", padding="same", name="block2_conv2")
        self.vgg19_block2_attention = CBAM(128, cbam_id='block2')
        self.vgg19_block2_pool = MaxPooling2D((2, 2), strides=(2, 2), name="block2_pool")

        # Block 3
        self.vgg19_block3_conv1 = Conv2D(256, (3, 3), activation="relu", padding="same", name="block3_conv1")
        self.vgg19_block3_conv2 = Conv2D(256, (3, 3), activation="relu", padding="same", name="block3_conv2")
        self.vgg19_block3_conv3 = Conv2D(256, (3, 3), activation="relu", padding="same", name="block3_conv3")
        self.vgg19_block3_conv4 = Conv2D(256, (3, 3), activation="relu", padding="same", name="block3_conv4")
        self.vgg19_block3_attention = CBAM(256, cbam_id='block3')
        self.vgg19_block3_pool = MaxPooling2D((2, 2), strides=(2, 2), name="block3_pool")
        
        # Block 4
        self.vgg19_block4_conv1 = Conv2D(512, (3, 3), activation="relu", padding="same", name="block4_conv1")
        self.vgg19_block4_conv2 = Conv2D(512, (3, 3), activation="relu", padding="same", name="block4_conv2")
        self.vgg19_block4_conv3 = Conv2D(512, (3, 3), activation="relu", padding="same", name="block4_conv3")
        self.vgg19_block4_conv4 = Conv2D(512, (3, 3), activation="relu", padding="same", name="block4_conv4")
        self.vgg19_block4_attention = CBAM(512, cbam_id='block4')
        self.vgg19_block4_pool = MaxPooling2D((2, 2), strides=(2, 2), name="block4_pool")

        # Block 5
        self.vgg19_block5_conv1 = Conv2D(512, (3, 3), activation="relu", padding="same", name="block5_conv1")
        self.vgg19_block5_conv2 = Conv2D(512, (3, 3), activation="relu", padding="same", name="block5_conv2")
        self.vgg19_block5_conv3 = Conv2D(512, (3, 3), activation="relu", padding="same", name="block5_conv3")
        self.vgg19_block5_conv4 = Conv2D(512, (3, 3), activation="relu", padding="same", name="block5_conv4")
        self.vgg19_block5_attention = CBAM(512, cbam_id='block5')
        self.vgg19_block5_pool = MaxPooling2D((2, 2), strides=(2, 2), name="block5_pool")

        # Classifier layer
        self.class_flatten = Flatten(name="class_flatten")
        
        self.class_dense1 = Dense(512, activation = 'relu', name="class_dense1")
        self.class_dropout1 = Dropout(0.4, name="class_dropout1")
        self.class_batch1 = BatchNormalization(name="class_batch1")
        
        self.class_dense2 = Dense(512, activation = 'relu', name="class_dense2")
        self.class_dropout2 = Dropout(0.3, name="class_dropout2")
        self.class_batch2 = BatchNormalization(name="class_batch2")

        self.class_dense3 = Dense(LABEL_CLASS, activation = 'softmax', name="class_dense3")

    def call(self, inputs, save_attention=False, training=False):
        # VGG19
        # Block 1
        x = self.vgg19_block1_conv1(inputs)
        x = self.vgg19_block1_conv2(x)
        x = self.vgg19_block1_attention(x, save_attention=save_attention)
        x = self.vgg19_block1_pool(x)

        # Block 2
        x = self.vgg19_block2_conv1(x)
        x = self.vgg19_block2_conv2(x)
        x = self.vgg19_block2_attention(x, save_attention=save_attention)
        x = self.vgg19_block2_pool(x)

        # Block 3
        x = self.vgg19_block3_conv1(x)
        x = self.vgg19_block3_conv2(x)
        x = self.vgg19_block3_conv3(x)
        x = self.vgg19_block3_conv4(x)
        x = self.vgg19_block3_attention(x, save_attention=save_attention)
        x = self.vgg19_block3_pool(x)

        # Block 4
        x = self.vgg19_block4_conv1(x)
        x = self.vgg19_block4_conv2(x)
        x = self.vgg19_block4_conv3(x)
        x = self.vgg19_block4_conv4(x)
        x = self.vgg19_block4_attention(x, save_attention=save_attention)
        x = self.vgg19_block4_pool(x)

        # Block 5
        x = self.vgg19_block5_conv1(x)
        x = self.vgg19_block5_conv2(x)
        x = self.vgg19_block5_conv3(x)
        x = self.vgg19_block5_conv4(x)
        x = self.vgg19_block5_attention(x, save_attention=save_attention)
        x = self.vgg19_block5_pool(x)

        # Classifier layer
        x = self.class_flatten(x)

        x = self.class_dense1(x)
        x = self.class_dropout1(x, training=training)
        x = self.class_batch1(x, training=training)

        x = self.class_dense2(x)
        x = self.class_dropout2(x, training=training)
        x = self.class_batch2(x, training=training)

        x = self.class_dense3(x)

        return x

    def build(self, input_shape):
        # VGG19
        # Block 1
        self.vgg19_block1_conv1.build(input_shape)
        x_shape = self.vgg19_block1_conv1.compute_output_shape(input_shape)
        self.vgg19_block1_conv2.build(x_shape)
        x_shape = self.vgg19_block1_conv2.compute_output_shape(x_shape)
        self.vgg19_block1_attention.build(x_shape)
        x_shape = self.vgg19_block1_attention.compute_output_shape(x_shape)
        self.vgg19_block1_pool.build(x_shape)
        x_shape = self.vgg19_block1_pool.compute_output_shape(x_shape)

        # Block 2
        self.vgg19_block2_conv1.build(x_shape)
        x_shape = self.vgg19_block2_conv1.compute_output_shape(x_shape)
        self.vgg19_block2_conv2.build(x_shape)
        x_shape = self.vgg19_block2_conv2.compute_output_shape(x_shape)
        self.vgg19_block2_attention.build(x_shape)
        x_shape = self.vgg19_block2_attention.compute_output_shape(x_shape)
        self.vgg19_block2_pool.build(x_shape)
        x_shape = self.vgg19_block2_pool.compute_output_shape(x_shape)

        # Block 3
        self.vgg19_block3_conv1.build(x_shape)
        x_shape = self.vgg19_block3_conv1.compute_output_shape(x_shape)
        self.vgg19_block3_conv2.build(x_shape)
        x_shape = self.vgg19_block3_conv2.compute_output_shape(x_shape)
        self.vgg19_block3_conv3.build(x_shape)
        x_shape = self.vgg19_block3_conv3.compute_output_shape(x_shape)
        self.vgg19_block3_conv4.build(x_shape)
        x_shape = self.vgg19_block3_conv4.compute_output_shape(x_shape)
        self.vgg19_block3_attention.build(x_shape)
        x_shape = self.vgg19_block3_attention.compute_output_shape(x_shape)
        self.vgg19_block3_pool.build(x_shape)
        x_shape = self.vgg19_block3_pool.compute_output_shape(x_shape)

        # Block 4
        self.vgg19_block4_conv1.build(x_shape)
        x_shape = self.vgg19_block4_conv1.compute_output_shape(x_shape)
        self.vgg19_block4_conv2.build(x_shape)
        x_shape = self.vgg19_block4_conv2.compute_output_shape(x_shape)
        self.vgg19_block4_conv3.build(x_shape)
        x_shape = self.vgg19_block4_conv3.compute_output_shape(x_shape)
        self.vgg19_block4_conv4.build(x_shape)
        x_shape = self.vgg19_block4_conv4.compute_output_shape(x_shape)
        self.vgg19_block4_attention.build(x_shape)
        x_shape = self.vgg19_block4_attention.compute_output_shape(x_shape)
        self.vgg19_block4_pool.build(x_shape)
        x_shape = self.vgg19_block4_pool.compute_output_shape(x_shape)

        # Block 5
        self.vgg19_block5_conv1.build(x_shape)
        x_shape = self.vgg19_block5_conv1.compute_output_shape(x_shape)
        self.vgg19_block5_conv2.build(x_shape)
        x_shape = self.vgg19_block5_conv2.compute_output_shape(x_shape)
        self.vgg19_block5_conv3.build(x_shape)
        x_shape = self.vgg19_block5_conv3.compute_output_shape(x_shape)
        self.vgg19_block5_conv4.build(x_shape)
        x_shape = self.vgg19_block5_conv4.compute_output_shape(x_shape)
        self.vgg19_block5_attention.build(x_shape)
        x_shape = self.vgg19_block5_attention.compute_output_shape(x_shape)
        self.vgg19_block5_pool.build(x_shape)
        x_shape = self.vgg19_block5_pool.compute_output_shape(x_shape)

        # Classifier layer
        self.class_flatten.build(x_shape)
        x_shape = self.class_flatten.compute_output_shape(x_shape)
        self.class_dense1.build(x_shape)
        x_shape = self.class_dense1.compute_output_shape(x_shape)
        self.class_dropout1.build(x_shape)
        x_shape = self.class_dropout1.compute_output_shape(x_shape)
        self.class_batch1.build(x_shape)
        x_shape = self.class_batch1.compute_output_shape(x_shape)
        
        self.class_dense2.build(x_shape)
        x_shape = self.class_dense2.compute_output_shape(x_shape)
        self.class_dropout2.build(x_shape)
        x_shape = self.class_dropout2.compute_output_shape(x_shape)
        self.class_batch2.build(x_shape)
        x_shape = self.class_batch2.compute_output_shape(x_shape)

        self.class_dense3.build(x_shape)


# FUNCTIONS
def Baseline_functional_model(num_classes, image_height, image_width, image_channels):
    # Input layer
    inputs = Input(shape=(image_width, image_height, image_channels), name='input_layer')

    x = Conv2D(64, (3, 3), activation="relu", padding="same", name="block1_conv1")(inputs)
    x = Conv2D(64, (3, 3), activation="relu", padding="same", name="block1_conv2")(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name="block1_pool")(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation="relu", padding="same", name="block2_conv1")(x)
    x = Conv2D(128, (3, 3), activation="relu", padding="same", name="block2_conv2")(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name="block2_pool")(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation="relu", padding="same", name="block3_conv1")(x)
    x = Conv2D(256, (3, 3), activation="relu", padding="same", name="block3_conv2")(x)
    x = Conv2D(256, (3, 3), activation="relu", padding="same", name="block3_conv3")(x)
    x = Conv2D(256, (3, 3), activation="relu", padding="same", name="block3_conv4")(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name="block3_pool")(x)
        
    # Block 4
    x = Conv2D(512, (3, 3), activation="relu", padding="same", name="block4_conv1")(x)
    x = Conv2D(512, (3, 3), activation="relu", padding="same", name="block4_conv2")(x)
    x = Conv2D(512, (3, 3), activation="relu", padding="same", name="block4_conv3")(x)
    x = Conv2D(512, (3, 3), activation="relu", padding="same", name="block4_conv4")(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name="block4_pool")(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation="relu", padding="same", name="block5_conv1")(x)
    x = Conv2D(512, (3, 3), activation="relu", padding="same", name="block5_conv2")(x)
    x = Conv2D(512, (3, 3), activation="relu", padding="same", name="block5_conv3")(x)
    x = Conv2D(512, (3, 3), activation="relu", padding="same", name="block5_conv4")(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name="block5_pool")(x)

    # Classifier layer
    x = Flatten(name="class_flatten")(x)
        
    x = Dense(512, activation = 'relu', name="class_dense1")(x)
    x = Dropout(0.4, name="class_dropout1")(x)
    x = BatchNormalization(name="class_batch1")(x)
        
    x = Dense(512, activation = 'relu', name="class_dense2")(x)
    x = Dropout(0.3, name="class_dropout2")(x)
    x = BatchNormalization(name="class_batch2")(x)

    outputs = Dense(26, activation = 'softmax', name="class_dense3")(x)

    return Model(inputs=inputs, outputs=outputs, name="functional_baseline")

    
def Se_functional_model(num_classes, image_height, image_width, image_channels):
    # Input layer
    inputs = Input(shape=(image_width, image_height, image_channels), name='input_layer')

    x = Conv2D(64, (3, 3), activation="relu", padding="same", name="block1_conv1")(inputs)
    x = Conv2D(64, (3, 3), activation="relu", padding="same", name="block1_conv2")(x)
    x = Squeeze_Excite(64, se_id='block1')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name="block1_pool")(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation="relu", padding="same", name="block2_conv1")(x)
    x = Conv2D(128, (3, 3), activation="relu", padding="same", name="block2_conv2")(x)
    x = Squeeze_Excite(128, se_id='block2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name="block2_pool")(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation="relu", padding="same", name="block3_conv1")(x)
    x = Conv2D(256, (3, 3), activation="relu", padding="same", name="block3_conv2")(x)
    x = Conv2D(256, (3, 3), activation="relu", padding="same", name="block3_conv3")(x)
    x = Conv2D(256, (3, 3), activation="relu", padding="same", name="block3_conv4")(x)
    x = Squeeze_Excite(256, se_id='block3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name="block3_pool")(x)
        
    # Block 4
    x = Conv2D(512, (3, 3), activation="relu", padding="same", name="block4_conv1")(x)
    x = Conv2D(512, (3, 3), activation="relu", padding="same", name="block4_conv2")(x)
    x = Conv2D(512, (3, 3), activation="relu", padding="same", name="block4_conv3")(x)
    x = Conv2D(512, (3, 3), activation="relu", padding="same", name="block4_conv4")(x)
    x = Squeeze_Excite(512, se_id='block4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name="block4_pool")(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation="relu", padding="same", name="block5_conv1")(x)
    x = Conv2D(512, (3, 3), activation="relu", padding="same", name="block5_conv2")(x)
    x = Conv2D(512, (3, 3), activation="relu", padding="same", name="block5_conv3")(x)
    x = Conv2D(512, (3, 3), activation="relu", padding="same", name="block5_conv4")(x)
    x = Squeeze_Excite(512, se_id='block5')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name="block5_pool")(x)

    # Classifier layer
    x = Flatten(name="class_flatten")(x)
        
    x = Dense(512, activation = 'relu', name="class_dense1")(x)
    x = Dropout(0.4, name="class_dropout1")(x)
    x = BatchNormalization(name="class_batch1")(x)
        
    x = Dense(512, activation = 'relu', name="class_dense2")(x)
    x = Dropout(0.3, name="class_dropout2")(x)
    x = BatchNormalization(name="class_batch2")(x)

    outputs = Dense(26, activation = 'softmax', name="class_dense3")(x)

    return Model(inputs=inputs, outputs=outputs, name="functional_se")


def Cbam_functional_model(num_classes, image_height, image_width, image_channels):
    # Input layer
    inputs = Input(shape=(image_width, image_height, image_channels), name='input_layer')

    x = Conv2D(64, (3, 3), activation="relu", padding="same", name="block1_conv1")(inputs)
    x = Conv2D(64, (3, 3), activation="relu", padding="same", name="block1_conv2")(x)
    x = CBAM(64, cbam_id='block1')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name="block1_pool")(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation="relu", padding="same", name="block2_conv1")(x)
    x = Conv2D(128, (3, 3), activation="relu", padding="same", name="block2_conv2")(x)
    x = CBAM(128, cbam_id='block2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name="block2_pool")(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation="relu", padding="same", name="block3_conv1")(x)
    x = Conv2D(256, (3, 3), activation="relu", padding="same", name="block3_conv2")(x)
    x = Conv2D(256, (3, 3), activation="relu", padding="same", name="block3_conv3")(x)
    x = Conv2D(256, (3, 3), activation="relu", padding="same", name="block3_conv4")(x)
    x = CBAM(256, cbam_id='block3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name="block3_pool")(x)
        
    # Block 4
    x = Conv2D(512, (3, 3), activation="relu", padding="same", name="block4_conv1")(x)
    x = Conv2D(512, (3, 3), activation="relu", padding="same", name="block4_conv2")(x)
    x = Conv2D(512, (3, 3), activation="relu", padding="same", name="block4_conv3")(x)
    x = Conv2D(512, (3, 3), activation="relu", padding="same", name="block4_conv4")(x)
    x = CBAM(512, cbam_id='block4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name="block4_pool")(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation="relu", padding="same", name="block5_conv1")(x)
    x = Conv2D(512, (3, 3), activation="relu", padding="same", name="block5_conv2")(x)
    x = Conv2D(512, (3, 3), activation="relu", padding="same", name="block5_conv3")(x)
    x = Conv2D(512, (3, 3), activation="relu", padding="same", name="block5_conv4")(x)
    x = CBAM(512, cbam_id='block5')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name="block5_pool")(x)

    # Classifier layer
    x = Flatten(name="class_flatten")(x)
        
    x = Dense(512, activation = 'relu', name="class_dense1")(x)
    x = Dropout(0.4, name="class_dropout1")(x)
    x = BatchNormalization(name="class_batch1")(x)
        
    x = Dense(512, activation = 'relu', name="class_dense2")(x)
    x = Dropout(0.3, name="class_dropout2")(x)
    x = BatchNormalization(name="class_batch2")(x)

    outputs = Dense(26, activation = 'softmax', name="class_dense3")(x)

    return Model(inputs=inputs, outputs=outputs, name="functional_cbam")