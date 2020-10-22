import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from network_configure import conf_basic_ops


"""This script defines basic operaters.
"""


def convolution_2D(inputs, filters, kernel_size, strides, use_bias, name=None):
    """Performs 2D convolution without activation function.
    If followed by batch normalization, set use_bias=False.
    """
    return tf.layers.conv2d(
                inputs=inputs,
                filters=filters,
                kernel_size=kernel_size,
                strides=strides,
                padding='same',
                use_bias=use_bias,
                kernel_initializer=conf_basic_ops['kernel_initializer'],
                name=name,
            )

def convolution_3D(inputs, filters, kernel_size, strides, use_bias, name=None):
    """Performs 3D convolution without activation function.
    If followed by batch normalization, set use_bias=False.
    """
    return tf.layers.conv3d(
                inputs=inputs,
                filters=filters,
                kernel_size=kernel_size,
                strides=strides,
                padding='same',
                use_bias=use_bias,
                kernel_initializer=conf_basic_ops['kernel_initializer'],
                name=name,
            )

def transposed_convolution_2D(inputs, filters, kernel_size, strides, use_bias, name=None):
    """Performs 2D transposed convolution without activation function.
    If followed by batch normalization, set use_bias=False.
    """
    return tf.layers.conv2d_transpose(
                inputs=inputs,
                filters=filters,
                kernel_size=kernel_size,
                strides=strides,
                padding='same',
                use_bias=use_bias,
                kernel_initializer=conf_basic_ops['kernel_initializer'],
                name=name,
            )

def transposed_convolution_3D(inputs, filters, kernel_size, strides, use_bias, name=None):
    """Performs 3D transposed convolution without activation function.
    If followed by batch normalization, set use_bias=False.
    """
    return tf.layers.conv3d_transpose(
                inputs=inputs,
                filters=filters,
                kernel_size=kernel_size,
                strides=strides,
                padding='same',
                use_bias=use_bias,
                kernel_initializer=conf_basic_ops['kernel_initializer'],
                name=name,
            )

def batch_norm(inputs, training, name=None):
    """Performs a batch normalization.
    We set fused=True for a significant performance boost.
    See https://www.tensorflow.org/performance/performance_guide#common_fused_ops
    """
    return tf.layers.batch_normalization(
                inputs=inputs,
                momentum=conf_basic_ops['momentum'],
                epsilon=conf_basic_ops['epsilon'],
                center=True,
                scale=True,
                training=training, 
                fused=True,
                name=name,
            )

def relu(inputs, name=None):
    return tf.nn.relu(inputs, name=name) if conf_basic_ops['relu_type'] == 'relu' \
            else tf.nn.relu6(inputs, name=name)
