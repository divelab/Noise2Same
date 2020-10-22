import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from basic_ops import *
from resnet_module import *


"""This script generates the U-Net architecture according to conf_unet.
"""

class UNet(object):
    def __init__(self, conf_unet):
        self.depth = conf_unet['depth']
        self.dimension = conf_unet['dimension']
        self.first_output_filters = conf_unet['first_output_filters']
        self.encoding_block_sizes = conf_unet['encoding_block_sizes']
        self.downsampling = conf_unet['downsampling']
        self.decoding_block_sizes = conf_unet['decoding_block_sizes']
        self.skip_method = conf_unet['skip_method']

    def __call__(self, inputs, training):
        """Add operations to classify a batch of input images.

        Args:
            inputs: A Tensor representing a batch of input images.
            training: A boolean. Set to True to add operations required only when
                training the classifier.

        Returns:
            A logits Tensor with shape [<batch_size>, self.num_classes].
        """

        return self._build_network(inputs, training)


    ################################################################################
    # Composite blocks building the network
    ################################################################################
    def _build_network(self, inputs, training):
        # first_convolution
        if self.dimension == '2D':
            convolution = convolution_2D
        elif self.dimension == '3D':
            convolution = convolution_3D
        inputs = convolution(inputs, self.first_output_filters, 3, 1, False, 'first_convolution')

        # encoding_block_1
        with tf.variable_scope('encoding_block_1'):
            for block_index in range(0, self.encoding_block_sizes[0]):
                inputs = res_block(inputs, self.first_output_filters, training, self.dimension,
                            'res_%d' % block_index)

        # encoding_block_i (down) = downsampling + zero or more res_block, i = 2, 3, ..., depth
        skip_inputs = [] # for identity skip connections
        for i in range(2, self.depth+1):
            skip_inputs.append(inputs)
            with tf.variable_scope('encoding_block_%d' % i):
                output_filters = self.first_output_filters * (2**(i-1))

                # downsampling
                downsampling_func = self._get_downsampling_function(self.downsampling[i-2])
                inputs = downsampling_func(inputs, output_filters, training, self.dimension,
                            'downsampling')

                for block_index in range(0, self.encoding_block_sizes[i-1]):
                    inputs = res_block(inputs, output_filters, training, self.dimension,
                                'res_%d' % block_index)

        # bottom_block = a combination of same_gto and res_block
        with tf.variable_scope('bottom_block'):
            output_filters = self.first_output_filters * (2**(self.depth-1))
            for block_index in range(0, 1):
                current_func = res_block
                inputs = current_func(inputs, output_filters, training, self.dimension,
                            'block_%d' % block_index)

        """
        Note: Identity skip connections are between the output of encoding_block_i and
        the output of upsampling in decoding_block_i, i = 1, 2, ..., depth-1.
        skip_inputs[i] is the output of encoding_block_i now.
        len(skip_inputs) == depth - 1
        skip_inputs[depth-2] should be combined during decoding_block_depth-1
        skip_inputs[0] should be combined during decoding_block_1
        """

        # decoding_block_j (up) = upsampling + zero or more res_block, j = depth-1, depth-2, ..., 1
        for j in range(self.depth-1, 0, -1):
            with tf.variable_scope('decoding_block_%d' % j):
                output_filters = self.first_output_filters * (2**(j-1))

                # upsampling
                upsampling_func = up_transposed_convolution
                inputs = upsampling_func(inputs, output_filters, training, self.dimension,
                            'upsampling')

                # combine with skip connections
                if self.skip_method == 'add':
                    inputs = tf.add(inputs, skip_inputs[j-1])
                elif self.skip_method == 'concat':
                    inputs = tf.concat([inputs, skip_inputs[j-1]], axis=-1)

                for block_index in range(0, self.decoding_block_sizes[self.depth-1-j]):
                    inputs = res_block(inputs, output_filters, training, self.dimension,
                                'res_%d' % block_index)

        return inputs


    def _get_downsampling_function(self, name):
        if name == 'down_res_block':
            return down_res_block
        elif name == 'convolution':
            return down_convolution
        else:
            raise ValueError("Unsupported function: %s." % (name))