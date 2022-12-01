import config
import tensorflow as tf
import models.efficientnet.lite.efficientnet_lite_builder as efficientnet_lite_builder
from models.Layerprovider import LayerProvider
from models.outputlayer import finallayerforoffsetoption
class EfficientNetLite0:

    def __init__(self, shape,is4Train=True,model_name = "efficientnet-lite0", totalJoints=13,offset = config.offset):
        self.offset = offset
        tf.reset_default_graph()# 利用这个可清空default graph以及nodes

        lProvider = LayerProvider(is4Train)
        outputlayer = finallayerforoffsetoption()
        self.model_name = model_name
        self.inputImage = tf.placeholder(tf.float32, shape=shape, name='Image')

        # output = lProvider.convb(self.inputImage, 3, 3, adaptChannels(32), 2, "1-conv-32-2-1", relu=True)

        output , endpoints = efficientnet_lite_builder.build_model(images=self.inputImage, model_name=self.model_name, training=is4Train, fine_tuning=True, features_only=True)

        output = lProvider.convb(output, 1, 1, 1280, 1, "2-conv-1280-2-1", relu=True)
        self.output = outputlayer.fornetworks(output, totalJoints)

    def getInput(self):
        return self.inputImage

    def getOutput(self):
        return self.output


