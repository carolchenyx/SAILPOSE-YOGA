import tensorflow as tf
from Config import config_cmd as config
from src.trainer.trainer import Trainer
from src.models.posenet import  PoseNet
from src.models.posenetv2 import PoseNetv2
from src.models.posenetv3 import PoseNetv3
from src.models.hrglassnetv3 import HourGlassNet
from src.models.efficientnetlite0 import EfficientNetLite0
from src.models.efficientnetlite1 import EfficientNetLite1
from src.models.senet import Senetget
from src.models.shufflenet import Shufflenetget
from src.models.resnet18 import ResNet
from src.dataprovider.Gaudataprovider import GauDataProviderAdaptator
from src.models.posenetNX import PoseNetNX
from opt import opt
import os

exp_dir = os.path.join("Result/{}/{}".format(opt.modeloutputFile, opt.Model_folder_name))
class buildPoseNet(object):

    def __init__(self):
        tf.reset_default_graph()
        self.offsetset = opt.offset
        self.modelDir = opt.checkpoinsaveDir
        self.dataformat = config.dataformat
        self.config_list = [{'sparsity': 0.8, 'op_types': ['default']}]


    def build(self,dataTrainProvider,dataValProvider, time, inputsize, inputshape, checkpointFile, is4Train
              , model_type):

        if model_type == "hourglass":
            model = HourGlassNet(is4Train=is4Train)

            output = model.getOutput()

            print(output.get_shape())

            intermediate_out = model.getInterOut()

            inputImage = model.getInput()
            if self.offsetset == False:
                trainer = Trainer(inputImage, output, intermediate_out, dataTrainProvider, dataValProvider, self.modelDir,
                                     Trainer.posenetLoss_nooffset,inputsize,self.dataformat,self.offsetset,time)
            else:
                trainer = Trainer(inputImage, output, intermediate_out, dataTrainProvider, dataValProvider,
                                  self.modelDir,
                                  Trainer.posenetLoss, inputsize, self.dataformat, self.offsetset, time)
        else:
            if model_type == "mobilenetv1":
                model = PoseNet(inputshape,is4Train=is4Train)

            elif model_type == "mobilenetv3":
                model = PoseNetv3(inputshape,is4Train=is4Train)
            elif model_type == "efficientnet0":
                model = EfficientNetLite0(inputshape,is4Train = is4Train)
            elif model_type == "efficientnet1":
                model = EfficientNetLite1(inputshape,is4Train = is4Train)
            elif model_type == "senet":
                model = Senetget(inputshape)
            elif model_type == "shufflenet":
                model = Shufflenetget(inputshape)
            elif model_type == "mobilenetv2":
                model = PoseNetv2(inputshape,is4Train=is4Train)
            elif model_type == "resnet18":
                model = ResNet(inputshape,is4Train=is4Train)
            elif model_type == "mobilenetXT":
                model = PoseNetNX(inputshape, is4Train=is4Train)
            else:
                print("wrong model")

            output = model.getOutput()
            layer = model.getlayers()
            print(model.getlayers())
            print(output.get_shape())

            inputImage = model.getInput()

            if opt.offset == False:
                trainer = Trainer(inputImage, output, [output], dataTrainProvider, dataValProvider, self.modelDir,
                                      Trainer.posenetLoss_nooffset,inputsize,self.dataformat,self.offsetset,time,layer)
            else:
                trainer = Trainer(inputImage, output, [output], dataTrainProvider, dataValProvider, self.modelDir,
                                      Trainer.posenetLoss,inputsize,self.dataformat,self.offsetset,time,layer)

        if not isinstance(checkpointFile, type(None)):
            trainer.restore(checkpointFile)

        return trainer



class train_pose(object):
    def __init__(self):
        self.bP =buildPoseNet()
        self.offsetset = opt.offset
        self.datanumber = config.datanumber
        self.batch = opt.batch
        self.trainanno = config.dataprovider_trainanno
        self.trainimg =config.dataprovider_trainimg
        self.testanno = config.dataprovider_testanno
        self.testimg = config.dataprovider_testimg
        self.dataTrainProvider = []
        self.datavalProvider = []


    def dataprovider(self,inputSize,outputSize):
        self.dataTrainProvider = []
        self.datavalProvider = []
        for i in range(self.datanumber):
            datatrain = GauDataProviderAdaptator(self.trainanno[i],self.trainimg[i],inputSize,outputSize,
                                                 self.batch,config.dataformat[i])
            datatest = GauDataProviderAdaptator(self.testanno[i], self.testimg[i], inputSize,outputSize,
                                                 self.batch, config.dataformat[i])
            self.dataTrainProvider.append(datatrain)
            self.datavalProvider.append(datatest)

        return self.dataTrainProvider,self.datavalProvider


    def train_pose(self, isTrain, checkpoints,
                       model, epochs, lrs, time_str, inputsize, outputsize, inputshape):

        dataTrainProvider, dataValProvider = self.dataprovider(inputsize,outputsize)
        posenet = self.bP.build(dataTrainProvider, dataValProvider,time_str, inputsize,inputshape,
                                checkpointFile=checkpoints,is4Train=isTrain, model_type=model)
        posenet.start(opt.fromStep, epochs, lrs, model, time_str)

        if model == "hourglass":
            self.export(posenet, outputName= exp_dir + "/" + model + str(isTrain)+"hourglass_out_3")

        else:
            if self.offsetset == True:
                self.export(posenet, outputFile= exp_dir+"/"+model + str(isTrain)+time_str + ".pb")
            else:
                self.export(posenet, outputFile=exp_dir+"/"+model + str(isTrain) + time_str + ".pb")


    def export(self,trainer, outputFile, outputName="Output"):#network/Output for resnet18

        input_graph_def = tf.get_default_graph().as_graph_def()

        output_graph_def = tf.graph_util.convert_variables_to_constants(
            trainer.sess,
            input_graph_def,
            [outputName]
        )

        with tf.gfile.GFile(outputFile, "wb") as f:
            f.write(output_graph_def.SerializeToString())