import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from basic_ops import *


"""This script defines non-attention same-, up-, down- modules.
Note that pre-activation is used for residual-like blocks.
Note that the residual block could be used for downsampling.
"""


def res_block(inputs, output_filters, training, dimension, name):
    """Standard residual block with pre-activation.

    Args:
        inputs: a Tensor with shape [batch, (d,) h, w, channels]
        output_filters: an integer
        training: a boolean for batch normalization and dropout
        dimension: a string, dimension of inputs/outputs -- 2D, 3D
        name: a string
        
    Returns:
        A Tensor of shape [batch, (_d,) _h, _w, output_filters]
    """
    if dimension == '2D':
        convolution = convolution_2D
        kernel_size = 3
    elif dimension == '3D':
        convolution = convolution_3D
        kernel_size = 3
    else:
        raise ValueError("Dimension (%s) must be 2D or 3D." % (dimension))

    with tf.variable_scope(name):
        if inputs.shape[-1] == output_filters:
            shortcut = inputs
            inputs = batch_norm(inputs, training, 'batch_norm_1')
            inputs = relu(inputs, 'relu_1')
        else:
            inputs = batch_norm(inputs, training, 'batch_norm_1')
            inputs = relu(inputs, 'relu_1')
            shortcut = convolution(inputs, output_filters, 1, 1, False, 'projection_shortcut')
        inputs = convolution(inputs, output_filters, kernel_size, 1, False, 'convolution_1')
        inputs = batch_norm(inputs, training, 'batch_norm_2')
        inputs = relu(inputs, 'relu_2')
        inputs = convolution(inputs, output_filters, kernel_size, 1, False, 'convolution_2')
        return tf.add(shortcut, inputs)


def down_res_block(inputs, output_filters, training, dimension, name):
    """Standard residual block with pre-activation for downsampling."""
    if dimension == '2D':
        convolution = convolution_2D
        projection_shortcut = convolution_2D
    elif dimension == '3D':
        convolution = convolution_3D
        projection_shortcut = convolution_3D
    else:
        raise ValueError("Dimension (%s) must be 2D or 3D." % (dimension))

    with tf.variable_scope(name):
        # The projection_shortcut should come after the first batch norm and ReLU.
        inputs = batch_norm(inputs, training, 'batch_norm_1')
        inputs = relu(inputs, 'relu_1')
        shortcut = projection_shortcut(inputs, output_filters, 1, 2, False, 'projection_shortcut')
        inputs = convolution(inputs, output_filters, 2, 2, False, 'convolution_1')
        inputs = batch_norm(inputs, training, 'batch_norm_2')
        inputs = relu(inputs, 'relu_2')
        inputs = convolution(inputs, output_filters, 3, 1, False, 'convolution_2')
        return tf.add(shortcut, inputs)

def down_convolution(inputs, output_filters, training, dimension, name):
    """Use a single stride 2 convolution for downsampling."""
    if dimension == '2D':
        convolution = convolution_2D
        pool = tf.layers.max_pooling2d
    elif dimension == '3D':
        convolution = convolution_3D
        pool = tf.layers.max_pooling3d
    else:
        raise ValueError("Dimension (%s) must be 2D or 3D." % (dimension))

    with tf.variable_scope(name):
        inputs = convolution(inputs, output_filters, 2, 2, True, 'convolution')
        return inputs

def up_transposed_convolution(inputs, output_filters, training, dimension, name):
    """Use a single stride 2 transposed convolution for upsampling."""
    if dimension == '2D':
        transposed_convolution = transposed_convolution_2D
    elif dimension == '3D':
        transposed_convolution = transposed_convolution_3D
    else:
        raise ValueError("Dimension (%s) must be 2D or 3D." % (dimension))

    with tf.variable_scope(name):
        inputs = transposed_convolution(inputs, output_filters, 2, 2, True, 'transposed_convolution')
        return inputs
