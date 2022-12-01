from src.utils.drawer import Drawer
from src.utils.bbox import BBox
from src.utils.interface import Pose2DInterface
from src.utils.LR import exponential_decay, polynomial_decay, inverse_time_decay
from tqdm import tqdm
from statistics import mean
import numpy as np
import os
import time
import tensorflow as tf
from opt import opt
from src.utils.loss import wing_loss
from src.utils.loss import adaptivewingLoss
from src.utils.loss import smooth_l1_loss
from Config import config_cmd as config
from src.utils.AUC_eval import AUC
from src.utils.Pose_eval import pose_eval
from src.utils.pose import Pose2D
from src.utils.Earlystop import earlystop
import cv2



os.environ["TF_MIN_GPU_MULTIPROCESSOR_COUNT"] = "4"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"


exp_dir = os.path.join("Result/{}/{}".format(opt.modeloutputFile, opt.Model_folder_name))
auc = AUC()
pose_eval = pose_eval()


class Trainer:
    SAVE_EVERY = opt.SAVE_EVERY
    TEST_EVERY = opt.TEST_EVERY
    VIZ_EVERY = opt.VIZ_EVERY
    num = config.datanumber

    def __init__(self, inputImage, output, outputStages, dataTrainProvider, dataValProvider, modelDir, lossFunc,
                 inputSize, datatpye, offsetset, time, layers, sess=None):
        self.inputSize = inputSize
        self.dataTrainProvider, self.dataValProvider = dataTrainProvider, dataValProvider
        self.inputImage = inputImage
        self.output = output
        self.offsetornot = offsetset
        self.dataformat = datatpye
        self.time = time
        self.layers = layers
        self.interation = 0
        if self.offsetornot == True:
            self.heatmapGT = tf.placeholder(tf.float32,
                                            shape=(None, output.shape[1], output.shape[2], opt.totaljoints * 3),
                                            name='heatmapGT')
        else:
            self.heatmapGT = tf.placeholder(tf.float32, shape=(None, output.shape[1], output.shape[2], opt.totaljoints),
                                            name='heatmapGT')

        self.globalStep = tf.Variable(0, trainable=False)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.trainLoss = []

        self.updater = []
        self.sess = tf.Session(config=config) if isinstance(sess, type(None)) else sess
        self.learningRate = tf.placeholder(tf.float32, [], name='learningRate')
        if opt.lr_type == "exponential_decay":
            self.lr = tf.train.exponential_decay(self.learningRate, global_step=self.globalStep,
                                                 decay_steps=opt.decay_steps, decay_rate=opt.decay_rate, staircase=True)
        elif opt.lr_type == "cosine_decay":
            self.lr = tf.train.cosine_decay(self.learningRate, global_step=self.globalStep,
                                            decay_steps=opt.decay_steps, alpha=0.0, name=None)
        elif opt.lr_type == "inverse_time_decay":
            self.lr = tf.train.inverse_time_decay(self.learningRate, global_step=self.globalStep,
                                                  decay_steps=opt.decay_steps, decay_rate=opt.decay_rate,
                                                  staircase=False, name=None)
        elif opt.lr_type == "polynomial_decay":
            self.lr = tf.train.polynomial_decay(self.learningRate, global_step=self.globalStep,
                                                decay_steps=opt.decay_steps,
                                                power=1.0, cycle=False, name=None)
        else:
            raise ValueError("Your lr_type name is wrong")

        if opt.optimizer == "Adam":
            self.opt = tf.train.AdamOptimizer(self.lr, epsilon=opt.epsilon)
        elif opt.optimizer == "Momentum":  # use_locking: 为True时锁定更新
            self.opt = tf.train.MomentumOptimizer(self.lr, momentum=opt.momentum, use_locking=False, name='Momentum',
                                                  use_nesterov=False)
        elif opt.optimizer == "Gradient":
            self.opt = tf.train.GradientDescentOptimizer(self.lr,
                                                         use_locking=False, name='GrandientDescent')
        else:
            raise ValueError("Your optimizer name is wrong")

        for i in range(len(self.dataTrainProvider)):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            lossall, Loss = self._buildLoss(self.heatmapGT, outputStages, dataTrainProvider[i], lossFunc,
                                            "trainLoss")
            self.trainLoss.append(Loss)

            self.grads = self.opt.compute_gradients(Loss)
            # self.apply_gradient_op = self.opt.apply_gradients(self.grads, global_step=self.globalStep)

            with tf.control_dependencies(update_ops):
                # self.train_op = tf.group(apply_gradient_op, variables_averages_op)
                upd = self.opt.minimize(self.trainLoss[i], self.globalStep)
            self.updater.append(upd)

        tf.summary.scalar("learningRate", self.lr)

        self.sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver(max_to_keep=50)
        for i in range(len(self.dataformat)):
            self.savePath = os.path.join(exp_dir, opt.backbone + "checkpoints" + self.time)


        self.fileWriter = tf.summary.FileWriter(
            os.path.join(exp_dir, opt.backbone + "logs_all" + self.time), self.sess.graph)
        self.summaryMerge = tf.summary.merge_all()

    def restore(self, checkpointPath):
        tf.train.Saver().restore(self.sess, checkpointPath)

    def setLearningRate(self, lr):
        self.sess.run(self.learningRate, feed_dict={self.learningRate: lr})

    def _buildLoss(self, heatmapGT, outputStages, data, lossFunc, lossName):
        batchSize = data.getBatchSize()
        jointweigt = data.getjoint_weight()
        usejointweight_ornot = data.getuse_different_joint_weights()
        losses = []
        lossesALL = []
        for idx, stage_out in enumerate(outputStages):
            loss, lossess = lossFunc(heatmapGT, stage_out, lossName + '_' + str(idx), batchSize, jointweigt,
                                     usejointweight_ornot)
            tf.summary.scalar(lossName + "_stage_" + str(idx), (tf.reduce_sum(loss) / batchSize))
            losses.append(loss)
            lossesALL.append(lossess)

        return lossesALL, (tf.reduce_sum(losses) / len(outputStages)) / batchSize

    @staticmethod
    def l2Loss(gt, pred, lossName, batchSize):
        return tf.nn.l2_loss(pred - gt, name=lossName)

    def posenetLoss_nooffset(gt, pred, lossName, batchSize):
        predHeat, gtHeat = pred[:, :, :, :opt.totaljoints], gt[:, :, :, :opt.totaljoints]
        totaljoints = opt.totaljoints

        if opt.hm_lossselect == 'l2':
            heatmapLoss = tf.nn.l2_loss(predHeat - gtHeat, name=lossName + "_heatmapLoss")
        elif opt.hm_lossselect == 'wing':
            heatmapLoss = wing_loss(predHeat, gtHeat)
        elif opt.hm_lossselect == 'adaptivewing':
            heatmapLoss = adaptivewingLoss(predHeat, gtHeat)
        elif opt.hm_lossselect == 'smooth_l1':
            heatmapLoss = smooth_l1_loss(None, predHeat, gtHeat)
        else:
            raise ValueError("Your optimizer name is wrong")

        for recordId in range(batchSize):
            for jointId in range(totaljoints):
                print(str(recordId) + "/" + str(batchSize) + " : " + str(jointId))
                # ================================> decode <x,y> from gt heatmap
                inlinedPix = tf.reshape(gtHeat[recordId, :, :, jointId], [-1])
                pixId = tf.argmax(inlinedPix)
                x = tf.floormod(pixId, gtHeat.shape[2])
                y = tf.cast(tf.divide(pixId, gtHeat.shape[2]), tf.int64)

        print("huber loss built")
        tf.summary.scalar(lossName + "_heatmapLoss", heatmapLoss)
        return heatmapLoss

    def posenetLoss(gt, pred, lossName, batchSize, jointweigt, usejointweight_ornot):
        predHeat, gtHeat = pred[:, :, :, :opt.totaljoints], gt[:, :, :, :opt.totaljoints]

        if usejointweight_ornot == True:
            target_weight = np.ones((1, 56, 56, opt.totaljoints),
                                    dtype=np.float32)
            target_weight = np.multiply(target_weight, jointweigt)
            predHeat = tf.multiply(predHeat, target_weight)
            gtHeat = tf.multiply(gtHeat, target_weight)

        predOffX, gtOffX = pred[:, :, :, opt.totaljoints:(2 * opt.totaljoints)], gt[:, :, :, opt.totaljoints:(
                2 * opt.totaljoints)]
        predOffY, gtOffY = pred[:, :, :, (2 * opt.totaljoints):], gt[:, :, :, (2 * opt.totaljoints):]
        totaljoints = opt.totaljoints

        if opt.hm_lossselect == 'l2':
            heatmapLoss = tf.nn.l2_loss(predHeat - gtHeat, name=lossName + "_heatmapLoss")
        elif opt.hm_lossselect == 'wing':
            heatmapLoss = wing_loss(predHeat, gtHeat)
        elif opt.hm_lossselect == 'adaptivewing':
            heatmapLoss = adaptivewingLoss(predHeat, gtHeat)
        elif opt.hm_lossselect == 'smooth_l1':
            heatmapLoss = smooth_l1_loss(None, predHeat, gtHeat)
        else:
            raise ValueError("Your optimizer name is wrong")
        offsetGT, offsetPred = [], []

        for recordId in range(batchSize):
            for jointId in range(totaljoints):
                print(str(recordId) + "/" + str(batchSize) + " : " + str(jointId))
                # ================================> decode <x,y> from gt heatmap

                inlinedPix = tf.reshape(gtHeat[recordId, :, :, jointId], [-1])
                pixId = tf.argmax(inlinedPix)

                x = tf.floormod(pixId, gtHeat.shape[2])
                y = tf.cast(tf.divide(pixId, gtHeat.shape[2]), tf.int64)

                # ==============================> add offset loss over the gt pix

                offsetGT.append(gtOffX[recordId, y, x, jointId])
                offsetPred.append(predOffX[recordId, y, x, jointId])
                offsetGT.append(gtOffY[recordId, y, x, jointId])
                offsetPred.append(predOffY[recordId, y, x, jointId])

        print("start building huber loss")
        offsetGT = tf.stack(offsetGT, 0)
        offsetPred = tf.stack(offsetPred, 0)
        offsetLoss = 5 * tf.losses.huber_loss(offsetGT, offsetPred)
        print("huber loss built")

        tf.summary.scalar(lossName + "_heatmapLoss", heatmapLoss)
        tf.summary.scalar(lossName + "_offsetLoss", offsetLoss)
        ht = predHeat - gtHeat

        return (heatmapLoss + offsetLoss), ht

    def _buildUpdater(self, loss, globalStep, lr):

        tf.summary.scalar("learningRate", lr)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            updater = tf.train.AdamOptimizer(lr, epsilon=1e-8).minimize(loss, globalStep)

        return updater

    def _toPose(self, output):

        totalJoints = opt.totaljoints

        if self.offsetornot == True:
            heatmap = output[:, :, :totalJoints]
            xOff = output[:, :, totalJoints:(totalJoints * 2)]
            yOff = output[:, :, (totalJoints * 2):]
        else:
            heatmap = output[:, :, :totalJoints]

        joints = np.zeros((totalJoints, 2)) - 1

        for jointId in range(totalJoints):
            inlinedPix = heatmap[:, :, jointId].reshape(-1)
            pixId = np.argmax(inlinedPix)

            outX = pixId % output.shape[1]
            outY = pixId // output.shape[1]
            if self.offsetornot == True:
                x = outX / output.shape[1] * self.inputImage.get_shape().as_list()[2] + xOff[outY, outX, jointId]
                y = outY / output.shape[0] * self.inputImage.get_shape().as_list()[1] + yOff[outY, outX, jointId]
            else:
                x = outX / output.shape[1] * self.inputImage.get_shape().as_list()[2]
                y = outY / output.shape[0] * self.inputImage.get_shape().as_list()[1]

            x = x / self.inputImage.get_shape().as_list()[2]
            y = y / self.inputImage.get_shape().as_list()[1]

            joints[jointId, 0] = x
            joints[jointId, 1] = y

        return Pose2D(joints)

    def _imageFeatureToImage(self, imageFeature):
        return (((imageFeature[:, :, :] + 1) / 2) * 255).astype(np.uint8)

    def _heatmapVisualisation(self, heatmaps):
        return ((heatmaps.sum(2) / heatmaps.sum(2).max()) * 255).astype(np.uint8)

    def average_gradients(self, tower_grads):
        """
        Get gradients of all variables.
        :param tower_grads:
        :return:
        """
        average_grads = []

        # get variable and gradients in differents gpus
        for grad_and_vars in zip(*tower_grads):
            # calculate the average gradient of each gpu
            grads = []
            for g, _ in grad_and_vars:
                expanded_g = tf.expand_dims(g, 0)
                grads.append(expanded_g)
            grad = tf.concat(grads, 0)
            grad = tf.reduce_mean(grad, 0)

            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        return average_grads
    def start(self, fromStep, totalSteps, lr, modeltype, date):
        total_iterations = 0
        best_validation_accuracy = 0.7
        last_improvement = 0  # 上一次有所改进的轮次
        cur_lr = lr
        j_num = 0
        require_improvement = opt.require_improvement  # 如果在1000轮内没有改进，停止迭代
        PCK = 0

        result = open(os.path.join(exp_dir, opt.backbone + date + "_result.csv"), "w")
        if opt.Early_stopping == False:
            result.write(
                "Backbone,Modelchannel0,Modelchannel1,Modelchannel2,Modelchannel3,Modelchannel4,Modelchannel5,"
                "isTrain,checkpoints_file,offset,inputsize,outputsize,optimizer,opt_epilon,"
                "momentum,heatmaploss, epsilon_loss, loss_w,Gauthreshold, GauSigma,depth_multiplier, "
                "datasetnumber,Dataset,Totaljoints,epochs,lr_type,decayrate,learning-rate,"
                "training_time,train_loss, PCK, PCKH ,auc_all,head,pckh_head,"
                "auc_head,lShoulder,pckh_lShoulder,auc_lShoulder, rShoulder,pckh_rShoulder,auc_rShoulder, "
                "lElbow,pckh_lElbow,auc_lElbow,rElbow,pckh_rElbow,auc_rElbow,"
                " lWrist,pckh_rWrist,auc_lWrist, rWrist,pckh_rWrist,auc_rWrist, "
                "lHip,pckh_lHip,auc_lHip,rHip,pckh_rHip,auc_rHip, "
                "lKnee,pckh_lKnee,auc_lKnee, "
                "rKnee,pckh_rKnee,auc_rKnee, "
                "lAnkle,pckh_lAnkle,auc_lAnkle,"
                "rAnkle,pckh_rAnkle,auc_rAnkle,"
                "traindata\n")
            result.close()

        if not os.path.exists("Result/{}/training_result.csv".format(opt.modeloutputFile)):
            result_all = open(os.path.join(opt.train_all_result, "training_result.csv"), "w")
            result_all.write(
                "Index,Backbone,Modelchannel0,Modelchannel1,Modelchannel2,Modelchannel3,Modelchannel4,Modelchannel5,isTrain,checkpoints_file,offset,inputsize,outputsize,optimizer,opt_epilon,momentum,heatmaploss, epsilon_loss, "
                "loss_w,Gauthreshold, GauSigma,depth_multiplier, datasetnumber,Batch,Dataset,Totaljoints,Total_epochs,Stop_epoch, learning_type,learning-rate,decay_rate,require_improvement, "
                "j_min,j_max,test_epoch,training_time,train_loss, best_validation_accuracy, Dataset,traindata\n")
            result_all.close()
        result_all = open(os.path.join(opt.train_all_result, "training_result.csv"), "a+")
        try:
            for i in range(fromStep, fromStep + totalSteps + 1):
                start_time = time.time()
                total_iterations += 1

                result = open(os.path.join(exp_dir, opt.backbone + date + "_result.csv"), "a+")
                PCKH_all_train = []

                for j in range(config.datanumber):


                    for self.interation in tqdm(range(int(2/opt.batch)),ascii=True,desc="Training:Epoch:{}".format(i)):#self.dataTrainProvider[j].get_data_leng()

                        inputs, heatmaps = self.dataTrainProvider[j].drawn()
                        if_exist = heatmaps[1]
                        res = self.sess.run([self.output, self.trainLoss[j], self.updater[j], self.summaryMerge],
                                            feed_dict={self.inputImage: inputs, self.heatmapGT: heatmaps[0],
                                                       self.learningRate: lr})
                        self.interation += 1
                        training_time = time.time() - start_time
                        self.interation = 0
                        self.fileWriter.add_summary(res[3], i)
                        fullscreen_bbox = BBox(0, 1, 0, 1)
                        y_true = {}
                        y_pred = {}
                        for batch_id in range(inputs.shape[0]):
                            pose_gt, confidence_gt = Pose2DInterface.our_approach_postprocessing(heatmaps[0][batch_id, :, :, :],
                                                                                     fullscreen_bbox, self.inputSize)
                            pose_pred, confidence_pred = Pose2DInterface.our_approach_postprocessing(res[0][batch_id, :, :, :],
                                                                                       fullscreen_bbox, self.inputSize)
                            y_true[batch_id] = Pose2D.get_joints(pose_gt)
                            y_pred[batch_id] = Pose2D.get_joints(pose_pred)

                        PCKH_train = pose_eval.cal_pckh(y_pred, y_true, if_exist, opt.eval_thresh,opt.totaljoints) #for one batch
                        PCKH_all_train = np.sum([PCKH_all_train,PCKH_train],axis = 0) #for all batch
                    PCKH_train = PCKH_all_train/int(self.dataTrainProvider[j].get_data_leng()/opt.batch)  #/int(self.dataTrainProvider[j].get_data_leng()/opt.batch)


                    print(
                        "Model_Folder:{}|--Epoch:{}|--isTrain:{}|--Earlystop:{}|--Train Loss:{}|--PCKH_train:{}|--lr:{}".format(
                            str(opt.Model_folder_name), str(i), str(opt.isTrain), str(opt.Early_stopping), res[1],
                            str(PCKH_train[-1]), str(cur_lr)))

                    summarypckh_train = tf.Summary(
                        value=[tf.Summary.Value(tag='train_PCKH', simple_value=PCKH_train[-1])]) #simple_value=mean(PCKH_train)

                train_loss = str(res[1])
                self.fileWriter.add_summary(summarypckh_train, i)

                if opt.Early_stopping:
                    if (total_iterations % opt.test_epoch == 0) or (i == totalSteps - 1):
                        if PCK > best_validation_accuracy:
                            best_validation_accuracy = PCK
                            last_improvement = total_iterations
                            j_num = 0
                            checkpoint_path = os.path.join(self.savePath, 'model')
                            self.saver.save(self.sess, checkpoint_path, global_step=i)

                        else:
                            lr, j_num = earlystop(cur_lr, j_num, i)

                    # 如果在require_improvement轮次内未有提升
                    if total_iterations - last_improvement > require_improvement or j_num > opt.j_max:
                        result_all.write(
                            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".
                                format(opt.Model_folder_name, str(config.modelchannel)[1:-1], modeltype, opt.isTrain,
                                       opt.checkpoints_file, opt.offset,
                                       self.inputSize[0], config.outputSize[0], opt.optimizer, opt.epsilon,
                                       opt.momentum,
                                       opt.hm_lossselect, opt.epsilon_loss, opt.w, opt.gaussian_thres,
                                       opt.gaussian_sigma, opt.depth_multiplier,
                                       config.datanumber, opt.batch, opt.dataset, opt.totaljoints, opt.epoch,
                                       total_iterations, opt.lr_type, lr, opt.decay_rate,
                                       opt.require_improvement, opt.j_min, opt.j_max, opt.test_epoch, training_time,
                                       train_loss, best_validation_accuracy,
                                       config.dataset_comment, str(config.dataformat)))
                        print("Stop optimization")
                        break

                else:
                    if i % Trainer.SAVE_EVERY == 0:
                        checkpoint_path = os.path.join(self.savePath, 'model')
                        self.saver.save(self.sess, checkpoint_path, global_step=i)

                PCKH_all_test =[]
                PCK_all_test=[]

                if i % Trainer.TEST_EVERY == 0:
                    print("Testing process...")

                    for self.interation in tqdm(range(int(2/ opt.testbatch)), ascii=True, desc="Testing"):#self.dataValProvider[j].get_data_leng()
                        inputs, heatmaps = self.dataValProvider[0].drawn_test()
                        if_exist = heatmaps[1]
                        res = self.sess.run([self.output, self.summaryMerge],
                                            feed_dict={self.inputImage: inputs, self.heatmapGT: heatmaps[0],
                                                       self.learningRate: 0})
                        fullscreen_bbox = BBox(0, 1, 0, 1)
                        y_true_test = {}
                        y_pred_tes = {}
                        for batch_id in range(inputs.shape[0]):
                            pose_gt, confidence_gt = Pose2DInterface.our_approach_postprocessing(
                                heatmaps[0][batch_id, :, :, :],
                                fullscreen_bbox, self.inputSize)
                            pose_pred, confidence_pred = Pose2DInterface.our_approach_postprocessing(
                                res[0][batch_id, :, :, :],
                                fullscreen_bbox, self.inputSize)
                            y_true_test[batch_id] = Pose2D.get_joints(pose_gt)
                            y_pred_tes[batch_id] = Pose2D.get_joints(pose_pred)
                            # pose_pred
                            # all labeled gt joints are used in the loss,
                            # if not detected by the prediction joint location (-1,-1) => (0.5,0.5)
                            tmp = pose_pred.get_joints()
                            tmp[~pose_pred.get_active_joints(), :] = 0.5

                            auc.auc_append(confidence_gt, confidence_pred)
                            pose_pred = Pose2D(tmp)
                            pcks, dist_KP, distances = pose_eval.save_value(pose_gt, pose_pred)

                        PCKH = pose_eval.cal_pckh(y_pred_tes, y_true_test, if_exist, opt.eval_thresh, opt.totaljoints)
                        PCK = np.sum(np.array(pcks)) / len(pcks)
                        PCK_all_test.append(PCK)
                        PCKH_all_test = np.sum([PCKH_all_test, PCKH], axis=0)
                    PCK = mean(PCK_all_test)
                    PCKH = PCKH_all_test/len(PCK_all_test)   # /len(PCK_all_test)
                    kps_acc = pose_eval.cal_eval()
                    auc_all = auc.auc_cal_all()
                    summarypck = tf.Summary(value=[tf.Summary.Value(tag="test_PCK", simple_value=PCK)])
                    summaryacc = tf.Summary(
                        value=[tf.Summary.Value(tag="testset_accuracy", simple_value=mean(distances))])
                    summarypckh = tf.Summary(value=[tf.Summary.Value(tag='test_PCKH', simple_value=PCKH[-1])])

                    print(
                        "Model_Folder:{}|--Epoch:{}|--Test_ACC:{}|--Test_PCK:{}|--Test_PCKH:{}|--Test_AUC:{}|--lr:{}".format(
                            str(opt.Model_folder_name), str(i), str(mean(distances))[:6],
                            str(PCK)[:6], str(PCKH[-1]), str(auc_all)[:6], str(cur_lr)))

                    auc_head, auc_leftShoulder, auc_rightShoulder, auc_leftElbow, auc_rightElbow, auc_leftWrist, \
                    auc_rightWrist, auc_leftHip, auc_rightHip, auc_leftKnee, auc_rightKnee, auc_leftAnkle, auc_rightAnkle = auc.auc_cal()
                    if opt.Early_stopping == False:
                        result.write(
                            "{},{},{},{},{},{},{},"
                            "{},{},{},{},{},{},{},"
                            "{},{},{},{},{},{},{},"
                            "{},{},{},{},{},{},{},"
                            "{},{},{},{},{},{},{},"
                            "{},{},{},{},{},{},{},"
                            "{},{},{},{},{},{},"
                            "{},{},{},{},{},{},"
                            "{},{},{},{},{},{},"
                            "{},{},{},"
                            "{},{},{},"
                            "{},{},{},"
                            "{},{},{},"
                            "{}\n".  # 73
                                format(modeltype, str(config.modelchannel)[0], str(config.modelchannel)[1],str(config.modelchannel)[2], str(config.modelchannel)[3],str(config.modelchannel)[4], str(config.modelchannel)[5],
                                       opt.isTrain, opt.checkpoints_file, opt.offset, self.inputSize[0], config.outputSize[0], opt.optimizer, opt.epsilon,
                                       opt.momentum, opt.hm_lossselect, opt.epsilon_loss, opt.w, opt.gaussian_thres,opt.gaussian_sigma, opt.depth_multiplier,
                                       config.datanumber, opt.dataset, opt.totaljoints, i, opt.lr_type, opt.decay_rate,cur_lr,
                                       training_time, train_loss, PCK, PCKH[-1], auc_all, kps_acc[0][0], PCKH[0] if opt.totaljoints == 16 else PCKH[9],
                                       auc_head, kps_acc[1][0], PCKH[1] if opt.totaljoints == 16 else PCKH[-3] , auc_leftShoulder, kps_acc[2][0], PCKH[2] if opt.totaljoints == 16 else PCKH[-4],auc_rightShoulder,
                                       kps_acc[3][0], PCKH[3] if opt.totaljoints == 16 else PCKH[-2], auc_leftElbow, kps_acc[4][0], PCKH[4] if opt.totaljoints == 16 else PCKH[-5], auc_rightElbow,
                                       kps_acc[5][0], PCKH[5] if opt.totaljoints == 16 else PCKH[-1], auc_leftWrist, kps_acc[6][0], PCKH[6] if opt.totaljoints == 16 else PCKH[-6], auc_rightWrist,
                                       kps_acc[7][0], PCKH[7] if opt.totaljoints == 16 else PCKH[3], auc_leftHip, kps_acc[8][0], PCKH[8] if opt.totaljoints == 16 else PCKH[2], auc_rightHip,
                                       kps_acc[9][0] if opt.totaljoints == 13 else 0, PCKH[9] if opt.totaljoints == 13  else PCKH[4] if opt.totaljoints == 16 else 0, auc_leftKnee,
                                       kps_acc[10][0] if opt.totaljoints == 13 else 0, PCKH[10] if opt.totaljoints == 13 else PCKH[1] if opt.totaljoints == 16 else 0, auc_rightKnee,
                                       kps_acc[11][0] if opt.totaljoints == 13 else 0, PCKH[11] if opt.totaljoints == 13 else PCKH[5] if opt.totaljoints == 16 else 0, auc_leftAnkle,
                                       kps_acc[12][0] if opt.totaljoints == 13 else 0, PCKH[12] if opt.totaljoints == 13 else PCKH[0] if opt.totaljoints == 16 else 0, auc_rightAnkle,
                                       str(config.dataformat)[1:-1]))
                    self.fileWriter.add_summary(summarypck, i)
                    self.fileWriter.add_summary(summaryacc, i)
                    self.fileWriter.add_summary(summarypckh, i)

                    if self.interation % Trainer.VIZ_EVERY == 0:
                        inputs_vis, heatmaps_vis = self.dataValProvider[0].drawn_visu()
                        res_vis = self.sess.run([self.output, self.trainLoss, self.summaryMerge],
                                                feed_dict={self.inputImage: inputs_vis,
                                                           self.heatmapGT: heatmaps_vis[0],
                                                           self.learningRate: 0})
                        gradients, variables = zip(*self.grads)
                        currHeatmaps = res_vis[0][0, :, :, :]
                        currImage = self._imageFeatureToImage(inputs_vis[0, :, :, :])
                        currHeatmapViz = self._heatmapVisualisation(currHeatmaps)
                        currHeatmapViz = currHeatmapViz.reshape(
                            (1, currHeatmapViz.shape[0], currHeatmapViz.shape[0], 1))
                        currPose = self._toPose(currHeatmaps)
                        skeletonViz = np.expand_dims(Drawer.draw_2d_pose(currImage, currPose), 0)

                        tmp = tf.summary.image("skeleton_" + str(i), skeletonViz).eval(session=self.sess)
                        self.fileWriter.add_summary(tmp, i)
                        tmp = tf.summary.image("heatmap_predicted_" + str(i), currHeatmapViz).eval(
                            session=self.sess)
                        self.fileWriter.add_summary(tmp, i)
                        with tf.name_scope('Feature' + str(i)):
                            for l in range(len(self.layers)):
                                if self.layers[l] == 'output':
                                    featureout =self.sess.graph.get_tensor_by_name("{}:0".format(self.layers[l]))

                                    if opt.backbone == 'mobilenetv2':
                                        featureall, grads_val = self.sess.run([featureout, gradients[-4]],  # 104/106
                                                                              feed_dict={self.inputImage: inputs,
                                                                                         self.heatmapGT: heatmaps[0],
                                                                                         self.learningRate: 0})
                                        weights = np.mean(grads_val)  # average pooling [13]
                                    if opt.backbone == 'mobilenetXT':
                                        featureall, grads_val = self.sess.run([featureout, gradients[-4]],  # 104
                                                                              feed_dict={self.inputImage: inputs,
                                                                                         self.heatmapGT: heatmaps,
                                                                                         self.learningRate: 0})
                                        weights = np.mean(grads_val)
                                    cam = np.ones(featureall.shape[1:3], dtype=np.float32)  # [56,56]
                                    # Taking a weighted average
                                    for i, w in enumerate(weights):
                                        cam += w * featureall[0, :, :, i]
                                    cam = np.maximum(cam, 0)
                                    cam = cam / np.max(cam)  # normalize
                                    cam3 = cv2.resize(cam, (224, 224))
                                    cam3 = cv2.applyColorMap(np.uint8(255 * cam3), cv2.COLORMAP_JET)
                                    cam3 = cv2.cvtColor(cam3, cv2.COLOR_BGR2RGB)
                                    img = cv2.resize(currImage, (224, 224))
                                    img = img.astype(float)
                                    img /= img.max()
                                    # Superimposing the visualization with the image.
                                    alpha = 0.0025
                                    new_img = img + alpha * cam3
                                    new_img /= new_img.max()

                                    tmp0 = tf.summary.image(str(self.layers[l]) + '-' + str(l),
                                                            np.expand_dims(new_img, 0)).eval(session=self.sess)

                                    self.fileWriter.add_summary(tmp0, l)

        except KeyboardInterrupt:
            result_all.write(
                "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".
                    format(opt.Model_folder_name, modeltype, str(config.modelchannel)[1:-1], opt.isTrain,
                           opt.checkpoints_file, opt.offset,
                           self.inputSize[0], config.outputSize[0], opt.optimizer, opt.epsilon, opt.momentum,
                           opt.hm_lossselect, opt.epsilon_loss, opt.w, opt.gaussian_thres, opt.gaussian_sigma,
                           opt.depth_multiplier,
                           config.datanumber, opt.batch, opt.dataset, opt.totaljoints, opt.epoch, total_iterations,
                           opt.lr_type, lr, opt.decay_rate, opt.require_improvement,
                           opt.j_min, opt.j_max, opt.test_epoch, training_time, train_loss, PCKH[-1],
                           config.dataset_comment, str(config.dataformat)[1:-1]))

            result_all.close()
        # else:
        #     result_all.write(
        #         "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".
        #             format(opt.Model_folder_name, modeltype, str(config.modelchannel)[1:-1], opt.isTrain,
        #                    opt.checkpoints_file, opt.offset,
        #                    self.inputSize[0], config.outputSize[0], opt.optimizer, opt.epsilon, opt.momentum,
        #                    opt.hm_lossselect, opt.epsilon_loss, opt.w, opt.gaussian_thres, opt.gaussian_sigma,
        #                    opt.depth_multiplier,
        #                    config.datanumber, opt.batch, opt.dataset, opt.totaljoints, opt.epoch, total_iterations,
        #                    opt.lr_type, lr, opt.decay_rate, opt.require_improvement,
        #                    opt.j_min, opt.j_max, opt.test_epoch, training_time, train_loss, PCKH[-1],
        #                    config.dataset_comment, str(config.dataformat)[1:-1]))
        #     result_all.close()