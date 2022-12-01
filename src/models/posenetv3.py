# -*- coding: utf-8 -*-

"""Implementation of Mobilenet V3.
Architecture: https://arxiv.org/pdf/1905.02244.pdf
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from src.models.outputlayer import finallayerforoffsetoption
from opt import opt
from src.models import Layerprovider



class PoseNetv3:
    def __init__(self, inputshape,is4Train , multiplier = 1, totalJoints = opt.totaljoints, size = opt.v3_version,  reuse = None):
        end_points = {}
        outputlayer = finallayerforoffsetoption()
        self.inputImage = tf.placeholder(tf.float32, shape=inputshape, name='Image')
        if size == 'small':
            layers = [
                [16, 16, 3, 2, "RE", False, 16],
                [16, 24, 3, 2, "RE", False, 72],
                [24, 24, 3, 1, "RE", False, 88],
                [24, 40, 5, 2, "RE", False, 96],
                [40, 40, 5, 1, "RE", False, 240],
                [40, 40, 5, 1, "RE", False, 240],
                [40, 48, 5, 1, "HS", True, 120],
                [48, 48, 5, 1, "HS", True, 144],
                [48, 96, 5, 2, "HS", True, 288],
                [96, 96, 5, 1, "HS", True, 576],
                [96, 96, 5, 1, "HS", True, 576],
            ]
        else:
            layers = [
                [16, 16, 3, 1, "RE", False, 16],
                [16, 24, 3, 2, "RE", False, 64],
                [24, 24, 3, 1, "RE", False, 72],
                [24, 40, 5, 2, "RE", True, 72],
                [40, 40, 5, 1, "RE", True, 120],

                [40, 40, 5, 1, "RE", True, 120],
                [40, 80, 3, 2, "HS", False, 240],
                [80, 80, 3, 1, "HS", False, 200],
                [80, 80, 3, 1, "HS", False, 184],
                [80, 80, 3, 1, "HS", False, 184],

                [80, 112, 3, 1, "HS", True, 480],
                [112, 112, 3, 1, "HS", True, 672],
                [112, 160, 5, 1, "HS", True, 672],
                [160, 160, 5, 2, "HS", True, 672],
                [160, 160, 5, 1, "HS", True, 960],
            ]
        input_size = self.inputImage.get_shape().as_list()[1:-1]
        assert ((input_size[0] % 32 == 0) and (input_size[1] % 32 == 0))

        reduction_ratio = 4
        with tf.variable_scope('init', reuse=reuse):
            init_conv_out = Layerprovider._make_divisible(16 * multiplier)
            x = Layerprovider._conv_bn_relu(self.inputImage, filters_num=init_conv_out, kernel_size=3, name='init',
                              use_bias=False, strides=2, is_training=is4Train, activation=Layerprovider.hard_swish)

        #with tf.variable_scope("MobilenetV3_small", reuse=reuse):
        for idx, (in_channels, out_channels, kernel_size, stride, activatation, se, exp_size) in enumerate(layers):
            in_channels = Layerprovider._make_divisible(in_channels * multiplier)
            out_channels = Layerprovider._make_divisible(out_channels * multiplier)
            exp_size = Layerprovider._make_divisible(exp_size * multiplier)
            x = Layerprovider.mobilenet_v3_block(x, kernel_size, exp_size, out_channels, stride,
                                   "bneck{}".format(idx), is_training=is4Train, use_bias=True,
                                   shortcut=(in_channels == out_channels), activatation=activatation,
                                   ratio=reduction_ratio, se=se)
            end_points["bneck{}".format(idx)] = x

        conv1_in = Layerprovider._make_divisible(96 * multiplier)
        conv1_out = Layerprovider._make_divisible(576 * multiplier)

        x = Layerprovider._conv_bn_relu(x, filters_num=conv1_out, kernel_size=1, name="conv1_out",
                          use_bias=True, strides=1, is_training=is4Train, activation=Layerprovider.hard_swish)
        # for DUC
        self.output = outputlayer.fornetworks_DUC(x, totalJoints)
        # for no DUC
        # self.output = outputlayer.fornetworks(output, totalJoints)

    def getInput(self):
        return self.inputImage

    def getOutput(self):
        return self.output



if __name__ == "__main__":
    print("begin ...")
