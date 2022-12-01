from src.models.Layerprovider1 import LayerProvider as LayerProvider1
from src.models.Layerprovider import LayerProvider
import tensorflow as tf
from opt import opt
from Config import config_cmd as config
import tensorflow.contrib.slim as slim


class finallayerforoffsetoption(object):
    def __init__(self, offset=opt.offset, pixel=config.pixelshuffle, conv=config.convb_13):
        self.lProvider1 = LayerProvider1(opt.isTrainpre)
        self.lProvider = LayerProvider(opt.isTrainpre)
        self.offset = offset
        self.pixel = pixel
        self.conv = conv

    def fornetworks_noDUC(self, output, totalJoints, layers):
        if opt.deconv == True:
            batch_size = tf.shape(output)[0]
            output = tf.nn.conv2d_transpose(output, filter=tf.Variable(tf.random.normal(shape=[4, 4, 320, 1280])),
                                            output_shape=[batch_size, 28, 28, 320],
                                            strides=[1, 2, 2, 1], padding='SAME',name='deconv-1280')
            layers.append(output.op.name)
            # output = tf.nn.conv2d_transpose(output, filter=tf.Variable(tf.random.normal(shape=[4, 4, 16, 1280])),
            #                                 output_shape=[batch_size, 112, 112, 16],
            #                                 strides=[1, 8, 8, 1], padding='SAME',name='deconv-1')

            # layers.append(output.op.name)
            output = tf.nn.conv2d_transpose(output, filter=tf.Variable(tf.random.normal(shape=[4, 4, 16, 320])),
                                            output_shape=[batch_size, 56, 56, 16],
                                            strides=[1, 2, 2, 1], padding='SAME',name='deconv-16')
            layers.append(output.op.name)
        else:
            output = tf.keras.layers.UpSampling2D(size=(2, 2),interpolation='bilinear')(output)
            layers.append(output.op.name)
            for i in range(len(self.pixel)):
                output = self.lProvider1.convb(output, 3, 3, 256, 1, config.convb_16[i][4], rate=2, relu=True)
            output = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(output)
            layers.append(output.op.name)
            output = self.lProvider1.convb(output, 3, 3, 16, 1, 'CON', rate=2, relu=True)
            layers.append(output.op.name)

        if self.offset == True:
            seg = self.lProvider1.pointwise_convolution(output, totalJoints, scope="output-1")
            layers.append(seg.op.name)
            seg = tf.sigmoid(seg, name=layers[-1] + "_sigmoid")
            layers.append(output.op.name)
            reg = self.lProvider1.pointwise_convolution(output, totalJoints * 2, scope="output-2")
            layers.append(reg.op.name)
            output = tf.concat([seg, reg], 3, name="Output")
            layers.append(output.op.name)
        else:
            output = self.lProvider1.pointwise_convolution(output, totalJoints, scope="output-1")
            layers.append(output.op.name)
            output = tf.identity(output, "Output")
            layers.append(output.op.name)
        return output, layers

    def fornetworks_DUC(self, output, totalJoints, layers):
        if opt.Shuffle == 2:
            if opt.backbone == "resnet18":
                output = tf.nn.depth_to_space(output, 2)  # pixel shuffle,name = 'depth_to_space_in_2'
                layers.append(output.op.name)
            else:
                output = tf.nn.depth_to_space(output, 2)  # pixel shuffle,name = 'depth_to_space_in_2'
                layers.append(output.op.name)
            for i in range(len(self.pixel)):
                if opt.totaljoints == 13:
                    output = self.lProvider1.convb(output, self.conv[i][0], self.conv[i][1], self.conv[i][2],
                                                   self.conv[i][3],
                                                   self.conv[i][4], relu=True)
                    layers.append(output.op.name)
                elif opt.totaljoints == 16:
                    output = self.lProvider1.convb(output, config.convb_16[i][0], config.convb_16[i][1],
                                                   config.convb_16[i][2], config.convb_16[i][3],
                                                   config.convb_16[i][4], relu=True)
                output = tf.nn.depth_to_space(output, self.pixel[i])  # name = 'depth_to_space_out_2'
                layers.append(output.op.name)
        elif opt.Shuffle == 4:
            if opt.backbone == "resnet18":
                output = tf.nn.depth_to_space(output, 2)  # pixel shuffle,name = 'depth_to_space_in_2'
                layers.append(output.op.name)
            else:
                output = tf.nn.depth_to_space(output, 2, name = 'depth_to_space_in_2')  # pixel shuffle,name = 'depth_to_space_in_2'
                layers.append(output.op.name)
            for i in range(len(self.pixel)):
                output = self.lProvider1.convb(output, config.convb_16[i][0], config.convb_16[i][1],
                                               config.convb_16[i][2], config.convb_16[i][3],
                                               config.convb_16[i][4], relu=True)
            output = tf.nn.depth_to_space(output, 4, name='output')
            layers.append(output.op.name)
        if self.offset == False:
            output = tf.identity(output, "Output")
            layers.append(output.op.name)
            # return a tensor with the same shape and contents as input.
        else:
            seg = self.lProvider1.pointwise_convolution(output, totalJoints, scope="output-1")
            layers.append(seg.op.name)
            seg = tf.sigmoid(seg)
            layers.append(output.op.name)
            reg = self.lProvider1.pointwise_convolution(output, totalJoints * 2, scope="output-2")
            layers.append(reg.op.name)
            output = tf.concat([seg, reg], 3, name="Output")
            layers.append(output.op.name)
            # output = tf.identity(output, "Output")
        return output, layers

    def _get_deconv_cfg(deconv_kernel):
        """Get configurations for deconv layers."""
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0
        else:
            raise ValueError(f'Not supported num_kernels ({deconv_kernel}).')

        return deconv_kernel, padding, output_padding
