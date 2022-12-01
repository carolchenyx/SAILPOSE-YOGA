
import tensorflow as tf

import src.models.efficientnet.lite.efficientnet_lite_builder as efficientnet_lite_builder
from src.models.outputlayer import finallayerforoffsetoption
class EfficientNetLite1:

    def __init__(self, shape,is4Train=True,model_name = "efficientnet-lite1", totalJoints=13):

        tf.reset_default_graph()# 利用这个可清空default graph以及nodes
        outputlayer = finallayerforoffsetoption()
        self.model_name = model_name
        self.inputImage = tf.placeholder(tf.float32, shape=shape, name='Image')

        output , endpoints = efficientnet_lite_builder.build_model(images=self.inputImage,
                                                                   model_name=self.model_name, training=is4Train, fine_tuning=True, features_only=True)
        self.output = outputlayer.fornetworks_DUC(output)

    def getInput(self):
        return self.inputImage

    def getOutput(self):
        return self.output


