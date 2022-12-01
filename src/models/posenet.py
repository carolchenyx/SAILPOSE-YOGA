import tensorflow as tf
from src.models.outputlayer import finallayerforoffsetoption
from opt import opt
if opt.activate_function == "swish":
    from src.models.Layerprovider1 import LayerProvider
elif opt.activate_function == "relu":
    from src.models.Layerprovider import LayerProvider

class PoseNet:

    def __init__(self, shape,is4Train=True, mobilenetVersion=0.75, totalJoints=opt.totaljoints, offset = opt.offset):
        self.offset = offset

        tf.reset_default_graph()# 利用这个可清空default graph以及nodes

        lProvider = LayerProvider(is4Train)
        outputlayer = finallayerforoffsetoption()

        adaptChannels = lambda totalLayer: int(mobilenetVersion * totalLayer)

        self.inputImage = tf.placeholder(tf.float32, shape=shape, name='Image')

        output = lProvider.convb(self.inputImage, 3, 3, adaptChannels(32), 2, "1-conv-32-2-1", relu=True)
        print("1-conv-32-2-1 : " + str(output.shape))

        # architecture description

        channels = [64, 128, 128, 256, 256, 512, 512, 512, 512, 512, 512, 1024, 1024]
        strides = [1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1]
        dilations = [1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 2]

        intermediateSupervision = [6]

        self.intermediateSupervisionOutputs = []

        for layerId in range(len(channels) - 1):

            filters = adaptChannels(channels[layerId])
            currStrides = strides[layerId]
            currDilation = dilations[layerId]

            layerDescription = str(layerId) + "-dw/pw-" + str(filters) + "-" + str(currStrides) + "-" + str(
                currDilation)

            output = lProvider.separable_conv(output, filters, 3, currStrides, dilation=currDilation,
                                              scope=layerDescription)

            print(layerDescription + " : " + str(output.shape))

            if layerId in intermediateSupervision:
                interSeg = lProvider.pointwise_convolution(output, totalJoints, scope=str(layerId) + "-inter-output-1")
                interSeg = tf.sigmoid(interSeg)
                interReg = lProvider.pointwise_convolution(output, totalJoints * 2,
                                                           scope=str(layerId) + "-inter-output-2")
                interOutput = tf.concat([interSeg, interReg], 3, name=str(layerId) + "-inter-output")
                self.intermediateSupervisionOutputs.append(interOutput)

        # last layer ===
        activationFunc = None
        filters = adaptChannels(channels[len(channels) - 1])
        currStrides = strides[len(channels) - 1]
        currDilation = dilations[len(channels) - 1]
        layerDescription = str(len(channels) - 1) + "-dw/pw-" + str(filters) + "-" + str(currStrides) + "-" + str(
            currDilation)

        output = lProvider.separable_conv(output, filters, 3, currStrides, dilation=currDilation, activationFunc=None,
                                          scope=layerDescription)
        #for DUC
        self.output = outputlayer.fornetworks_DUC(output,totalJoints)
        #for no DUC
        # self.output = outputlayer.fornetworks(output, totalJoints)

    def getInput(self):
        return self.inputImage

    def getIntermediateOutputs(self):
        return self.intermediateSupervisionOutputs[:]

    def getOutput(self):
        return self.output


# class LayerProvider:
#
#     def __init__(self, is4Train):
#
#         self.init_xavier = tf.contrib.layers.xavier_initializer()
#         self.init_norm = tf.truncated_normal_initializer(stddev=0.01)
#         self.init_zero = slim.init_ops.zeros_initializer()
#         self.l2_regularizer = tf.contrib.layers.l2_regularizer(0.00004)
#
#         self.is4Train = is4Train
#
#     def max_pool(self, inputs, k_h, k_w, s_h, s_w, name, padding="SAME"):
#         return tf.nn.max_pool(inputs,
#                               ksize=[1, k_h, k_w, 1],
#                               strides=[1, s_h, s_w, 1],
#                               padding=padding,
#                               name=name)
#
#     def upsample(self, inputs, shape, name):
#         return tf.image.resize_bilinear(inputs, shape, name=name)
#
#     def separable_conv(self, input, c_o, k_s, stride, dilation=1, activationFunc=tf.nn.relu6, scope=""):
#
#         with slim.arg_scope([slim.batch_norm],
#                             decay=0.999,
#                             fused=True,
#                             is_training=self.is4Train,
#                             activation_fn=activationFunc):
#             output = slim.separable_convolution2d(input,
#                                                   num_outputs=None,
#                                                   stride=stride,
#                                                   trainable=self.is4Train,
#                                                   depth_multiplier=1.0,
#                                                   kernel_size=[k_s, k_s],
#                                                   rate=dilation,
#                                                   weights_initializer=self.init_xavier,
#                                                   weights_regularizer=self.l2_regularizer,
#                                                   biases_initializer=None,
#                                                   activation_fn=tf.nn.relu6,
#                                                   scope=scope + '_depthwise')
#
#             output = slim.convolution2d(output,
#                                         c_o,
#                                         stride=1,
#                                         kernel_size=[1, 1],
#                                         weights_initializer=self.init_xavier,
#                                         biases_initializer=self.init_zero,
#                                         normalizer_fn=slim.batch_norm,
#                                         trainable=self.is4Train,
#                                         weights_regularizer=None,
#                                         scope=scope + '_pointwise')
#
#         return output
#
#     def pointwise_convolution(self, inputs, channels, scope=""):
#
#         with tf.variable_scope("merge_%s" % scope):
#             with slim.arg_scope([slim.batch_norm],
#                                 decay=0.999,
#                                 fused=True,
#                                 is_training=self.is4Train):
#                 return slim.convolution2d(inputs,
#                                           channels,
#                                           stride=1,
#                                           kernel_size=[1, 1],
#                                           activation_fn=None,
#                                           weights_initializer=self.init_xavier,
#                                           biases_initializer=self.init_zero,
#                                           normalizer_fn=slim.batch_norm,
#                                           weights_regularizer=None,
#                                           scope=scope + '_pointwise',
#                                           trainable=self.is4Train)
#
#     def inverted_bottleneck(self, inputs, up_channel_rate, channels, stride , k_s=3, dilation=1.0, scope=""):
#
#         with tf.variable_scope("inverted_bottleneck_%s" % scope):
#             with slim.arg_scope([slim.batch_norm],
#                                 decay=0.999,
#                                 fused=True,
#                                 is_training=self.is4Train):
#                 #stride = 2 if subsample else 1
#
#                 output = slim.convolution2d(inputs,
#                                             up_channel_rate * inputs.get_shape().as_list()[-1],
#                                             stride=1,
#                                             kernel_size=[1, 1],
#                                             weights_initializer=self.init_xavier,
#                                             biases_initializer=self.init_zero,
#                                             activation_fn=tf.nn.relu6,
#                                             normalizer_fn=slim.batch_norm,
#                                             weights_regularizer=None,
#                                             scope=scope + '_up_pointwise',
#                                             trainable=self.is4Train)
#
#                 output = slim.separable_convolution2d(output,
#                                                       num_outputs=None,
#                                                       stride=stride,
#                                                       depth_multiplier=1.0,
#                                                       activation_fn=tf.nn.relu6,
#                                                       kernel_size=k_s,
#                                                       weights_initializer=self.init_xavier,
#                                                       weights_regularizer=self.l2_regularizer,
#                                                       biases_initializer=None,
#                                                       normalizer_fn=slim.batch_norm,
#                                                       rate=dilation,
#                                                       padding="SAME",
#                                                       scope=scope + '_depthwise',
#                                                       trainable=self.is4Train)
#
#                 output = slim.convolution2d(output,
#                                             channels,
#                                             stride=1,
#                                             kernel_size=[1, 1],
#                                             activation_fn=None,
#                                             weights_initializer=self.init_xavier,
#                                             biases_initializer=self.init_zero,
#                                             normalizer_fn=slim.batch_norm,
#                                             weights_regularizer=None,
#                                             scope=scope + '_pointwise',
#                                             trainable=self.is4Train)
#
#                 if inputs.get_shape().as_list()[1:] == output.get_shape().as_list()[1:]:
#                     output = tf.add(inputs, output)
#         return output
#
#     def convb(self, input, k_h, k_w, c_o, stride, name, relu=True):
#
#         with slim.arg_scope([slim.batch_norm], decay=0.999, fused=True, is_training=self.is4Train):
#             output = slim.convolution2d(
#                 inputs=input,
#                 num_outputs=c_o,
#                 kernel_size=[k_h, k_w],
#                 stride=stride,
#                 normalizer_fn=slim.batch_norm,
#                 weights_regularizer=self.l2_regularizer,
#                 weights_initializer=self.init_xavier,
#                 biases_initializer=self.init_zero,
#                 activation_fn=tf.nn.relu if relu else None,
#                 scope=name,
#                 trainable=self.is4Train)
#
#         return output
#
#     def convb_nopadding(self, input, k_h, k_w, c_o, stride, name, relu=True):
#
#         with slim.arg_scope([slim.batch_norm], decay=0.999, fused=True, is_training=self.is4Train):
#             output = slim.convolution2d(
#                 inputs=input,
#                 num_outputs=c_o,
#                 kernel_size=[k_h, k_w],
#                 stride=stride,
#                 padding='VALID',
#                 normalizer_fn=slim.batch_norm,
#                 weights_regularizer=self.l2_regularizer,
#                 weights_initializer=self.init_xavier,
#                 biases_initializer=self.init_zero,
#                 activation_fn=tf.nn.relu if relu else None,
#                 scope=name,
#                 trainable=self.is4Train)
#
#         return output
#
#     def stage(self, inputs, outputSize, stageNumber, kernel_size=3):
#
#         output = slim.stack(inputs, self.inverted_bottleneck,
#                             [
#                                 (2, 32, 0, kernel_size, 4),
#                                 (2, 32, 0, kernel_size, 2),
#                                 (2, 32, 0, kernel_size, 1),
#                             ], scope="stage_%d_mv2" % stageNumber)
#
#         return slim.stack(output, self.separable_conv,
#                           [
#                               (64, 1, 1),
#                               (outputSize, 1, 1)
#                           ], scope="stage_%d_mv1" % stageNumber)
#
#     def get(self, input, layerDesc, name):
#
#         if layerDesc['op'] == 'conv2d':
#             return self.convb(input, 3, 3, layerDesc['outputSize'], layerDesc['stride'], name, relu=True)
#         elif layerDesc['op'] == 'bottleneck':
#             return self.inverted_bottleneck(input, layerDesc['expansion'], layerDesc['outputSize'],
#                                             layerDesc['stride'] == 2, k_s=3, dilation=layerDesc['dilation'], scope=name)
#         elif layerDesc['op'] == 'multi_scale_bottleneck':
#             return self.multi_scale_inverted_bottleneck(input, layerDesc['expansion'], layerDesc['outputSize'],
#                                                         layerDesc['stride'] == 2, k_s=3, scope=name)
#         else:
#             return None