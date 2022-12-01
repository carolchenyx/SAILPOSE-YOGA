from Config import config_cmd as config
import os
import time
from src.modelbuild.PoseBuild import train_pose
from opt import opt


time_str = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
os.makedirs("Result/{}/{}".format(opt.modeloutputFile, opt.Model_folder_name), exist_ok=True)
exp_dir = os.path.join("Result/{}/{}".format(opt.modeloutputFile, opt.Model_folder_name))


class run:
    def __init__(self):
        self.tp = train_pose()
        self.inputSize = config.inputSize
        self.outputSize = config.outputSize
        self.trainornot = opt.isTrain
        self.checkpoint = opt.checkpoints_file
        self.model = opt.backbone
        self.grayimage = opt.grayimage


    def train(self):
        self.tp.train_pose( self.trainornot, self.checkpoint, self.model,
                        opt.epoch, opt.lr, time_str, self.inputSize,self.outputSize,
                                config.inputshape)



if __name__ == '__main__':
    R = run()
    R.train()