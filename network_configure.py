import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf


"""This is the configuration file.
"""


################################################################################
# Settings for Basic Operaters
################################################################################

conf_basic_ops = dict()

# kernel_initializer for convolutions and transposed convolutions
# If None, the default initializer is the Glorot (Xavier) normal initializer.
conf_basic_ops['kernel_initializer'] = tf.glorot_uniform_initializer()

# momentum for batch normalization
conf_basic_ops['momentum'] = 0.997

# epsilon for batch normalization
conf_basic_ops['epsilon'] = 1e-5

# String options: 'relu', 'relu6'
conf_basic_ops['relu_type'] = 'relu'

################################################################################
# Settings for Attention Modules
################################################################################

# Set the attention in same_gto
conf_attn_same = dict()

# Define the relationship between total_key_filters and output_filters.
# total_key_filters = output_filters // key_ratio
conf_attn_same['key_ratio'] = 1

# Define the relationship between total_value_filters and output_filters.
# total_key_filters = output_filters // value_ratio
conf_attn_same['value_ratio'] = 1

# number of heads
conf_attn_same['num_heads'] = 2

# dropout rate, 0.0 means no dropout
conf_attn_same['dropout_rate'] = 0.0

# whether to use softmax on attention_weights
conf_attn_same['use_softmax'] = False

# whether to use bias terms in input/output transformations
conf_attn_same['use_bias'] = True

# Set the attention in up_gto
conf_attn_up = dict()

conf_attn_up['key_ratio'] = 1
conf_attn_up['value_ratio'] = 1
conf_attn_up['num_heads'] = 2
conf_attn_up['dropout_rate'] = 0
conf_attn_up['use_softmax'] = False
conf_attn_up['use_bias'] = True

# Set the attention in down_gto
conf_attn_down = dict()

conf_attn_down['key_ratio'] = 1
conf_attn_down['value_ratio'] = 1
conf_attn_down['num_heads'] = 2
conf_attn_down['dropout_rate'] = 0.0
conf_attn_down['use_softmax'] = False
conf_attn_down['use_bias'] = True

################################################################################
# Describing the U-net
################################################################################

conf_unet = dict()

"""
Describe your U-Net under the following framework:

********************************************************************************************
layers													|	output_filters
														|
first_convolution + encoding_block_1 (same)				|	first_output_filters
+ encoding_block_i, i = 2, 3, ..., depth. (down)		|	first_output_filters*(2**(i-1))
+ bottom_block											|	first_output_filters*(2**(depth-1))
+ decoding_block_j, j = depth-1, depth-2, ..., 1 (up)	|	first_output_filters*(2**(j-1))
+ output_layer
********************************************************************************************

Specifically,
encoding_block_1 (same) = one or more res_block
encoding_block_i (down) = downsampling + zero or more res_block, i = 2, 3, ..., depth-1
encoding_block_depth (down) = downsampling
bottom_block = a combination of same_gto and res_block
decoding_block_j (up) = upsampling + zero or more res_block, j = depth-1, depth-2, ..., 1

Identity skip connections are between the output of encoding_block_i and
the output of upsampling in decoding_block_i, i = 1, 2, ..., depth-1.
The combination method could be 'add' or 'concat'.
"""

# Set U-Net depth.
conf_unet['depth'] = 3

# Set the output_filters for first_convolution and encoding_block_1 (same).
conf_unet['first_output_filters'] = 96

# Set the encoding block sizes, i.e., number of res_block in encoding_block_i, i = 1, 2, ..., depth.
# It is an integer list whose length equals to depth.
# The first entry should be positive since encoding_block_1 = one or more res_block.
# The last entry should be zero since encoding_block_depth (down) = downsampling.
conf_unet['encoding_block_sizes'] = [1, 1, 0]

# Set the decoding block sizes, i.e., number of res_block in decoding_block_j, j = depth-1, depth-2, ..., 1.
# It is an integer list whose length equals to depth-1.
conf_unet['decoding_block_sizes'] = [1, 1]

# Set the downsampling methods for each encoding_block_i, i = 2, 3, ..., depth.
# It is an string list whose length equals to depth-1.
# String options: 'down_gto_v1', 'down_gto_v2', 'down_res_block', 'convolution'
conf_unet['downsampling'] = ['convolution', 'convolution']

# Set the combination method for identity skip connections
# String options: 'add', 'concat'
conf_unet['skip_method'] = 'concat'

# Set the output layer


# Check
assert conf_unet['depth'] == len(conf_unet['encoding_block_sizes'])
assert conf_unet['encoding_block_sizes'][0] > 0
assert conf_unet['encoding_block_sizes'][-1] == 0
assert conf_unet['depth'] == len(conf_unet['decoding_block_sizes']) + 1
assert conf_unet['depth'] == len(conf_unet['downsampling']) + 1
assert conf_unet['skip_method'] in ['add', 'concat']