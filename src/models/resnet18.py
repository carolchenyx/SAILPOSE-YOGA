from opt import opt
import tensorflow as tf
from src.models.common import resblock,bottle_resblock,get_residual_layer,\
    conv
from src.models.outputlayer import finallayerforoffsetoption
if opt.activate_function == "swish":
    from src.models.Layerprovider1 import LayerProvider
elif opt.activate_function == "relu":
    from src.models.Layerprovider import LayerProvider



class ResNet(object):
    def __init__(self, shape, is4Train, mobilenetVersion=1, totalJoints=opt.totaljoints):
        tf.reset_default_graph()  # 利用这个可清空default graph以及nodes
        self.inputImage = tf.placeholder(tf.float32, shape=shape, name='Image')
        self.transtrain = is4Train
        # self.istrain = is4Train
        self.model_name = 'ResNet'
        self.res_n = 18
        self.layers = []
        self.model_layers = 0
        outputlayer = finallayerforoffsetoption()
        lProvider = LayerProvider(is4Train)

    ##################################################################################
    # Generator
    ##################################################################################

        with tf.variable_scope("network", reuse=False):

            if self.res_n < 50:
                residual_block = resblock
            else:
                residual_block = bottle_resblock

            residual_list = get_residual_layer(self.res_n)

            ch = 32  # paper is 64
            # x = conv(self.inputImage, channels=ch, kernel=3, stride=1, scope='conv')
            x = lProvider.conv(self.inputImage, channels=ch, kernel=3, stride=1, scope='conv')
            self.layers.append(x.op.name)
            for i in range(residual_list[0]):
                x = residual_block(x, channels=ch, is_training=self.transtrain, downsample=False,
                                   scope='resblock0_' + str(i))
                self.layers.append(x.op.name)

            x = residual_block(x, channels=ch * 2, is_training=self.transtrain, downsample=True, scope='resblock1_0')
            self.layers.append(x.op.name)
            for i in range(1, residual_list[1]):
                x = residual_block(x, channels=ch * 2, is_training=self.transtrain, downsample=False,
                                   scope='resblock1_' + str(i))
                self.layers.append(x.op.name)


            x = residual_block(x, channels=ch * 4, is_training=self.transtrain, downsample=True, scope='resblock2_0')
            self.layers.append(x.op.name)
            for i in range(1, residual_list[2]):
                x = residual_block(x, channels=ch * 4, is_training=self.transtrain, downsample=False,
                                   scope='resblock2_' + str(i))
                self.layers.append(x.op.name)

            x = residual_block(x, channels=ch * 8, is_training=self.transtrain, downsample=True, scope='resblock_3_0')
            self.layers.append(x.op.name)
            for i in range(1, residual_list[3]):
                x = residual_block(x, channels=ch * 8, is_training=self.transtrain, downsample=False,
                                   scope='resblock_3_' + str(i))
                self.layers.append(x.op.name)


            self.output, self.model_layers = outputlayer.fornetworks_DUC(x, totalJoints,self.layers)

    def getOutput(self):
        return self.output

    def getInput(self):
        return self.inputImage

    def getlayers(self):
        return self.model_layers



