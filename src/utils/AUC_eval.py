from sklearn.metrics import roc_auc_score
from opt import opt
class AUC:

    def __init__(self):
        self.confidences_gt_all = []
        self.confidences_pred_all = []
        self.head_gt = []
        self.leftShoulder_gt = []
        self.rightShoulder_gt = []
        self.leftElbow_gt = []
        self.rightElbow_gt = []
        self.leftWrist_gt = []
        self.rightWrist_gt = []
        self.leftHip_gt = []
        self.rightHip_gt = []
        self.leftKnee_gt = []
        self.rightKnee_gt = []
        self.leftAnkle_gt = []
        self.rightAnkle_gt = []

        self.head_pred = []
        self.leftShoulder_pred = []
        self.rightShoulder_pred = []
        self.leftElbow_pred = []
        self.rightElbow_pred = []
        self.leftWrist_pred = []
        self.rightWrist_pred = []
        self.leftHip_pred = []
        self.rightHip_pred = []
        self.leftKnee_pred = []
        self.rightKnee_pred = []
        self.leftAnkle_pred = []
        self.rightAnkle_pred = []

    def auc_append(self,confidence_gt,confidence_pred,joints=opt.totaljoints):
        for item in confidence_gt:
            self.confidences_gt_all.append(item)
        for itempre in confidence_pred:
            self.confidences_pred_all.append(itempre)
        if joints == 13:
            self.head_gt.append(confidence_gt[0])
            self.leftShoulder_gt.append(confidence_gt[1])
            self.rightShoulder_gt.append(confidence_gt[2])
            self.leftElbow_gt.append(confidence_gt[3])
            self.rightElbow_gt.append(confidence_gt[4])
            self.leftWrist_gt.append(confidence_gt[5])
            self.rightWrist_gt.append(confidence_gt[6])
            self.leftHip_gt.append(confidence_gt[7])
            self.rightHip_gt.append(confidence_gt[8])
            self.leftKnee_gt.append(confidence_gt[9])
            self.rightKnee_gt.append(confidence_gt[10])
            self.leftAnkle_gt.append(confidence_gt[11])
            self.rightAnkle_gt.append(confidence_gt[12])

            self.head_pred.append(confidence_pred[0])
            self.leftShoulder_pred.append(confidence_pred[1])
            self.rightShoulder_pred.append(confidence_pred[2])
            self.leftElbow_pred.append(confidence_pred[3])
            self.rightElbow_pred.append(confidence_pred[4])
            self.leftWrist_pred.append(confidence_pred[5])
            self.rightWrist_pred.append(confidence_pred[6])
            self.leftHip_pred.append(confidence_pred[7])
            self.rightHip_pred.append(confidence_pred[8])
            self.leftKnee_pred.append(confidence_pred[9])
            self.rightKnee_pred.append(confidence_pred[10])
            self.leftAnkle_pred.append(confidence_pred[11])
            self.rightAnkle_pred.append(confidence_pred[12])
        elif joints == 9:
            self.head_gt.append(confidence_gt[0])
            self.leftShoulder_gt.append(confidence_gt[1])
            self.rightShoulder_gt.append(confidence_gt[2])
            self.leftElbow_gt.append(confidence_gt[3])
            self.rightElbow_gt.append(confidence_gt[4])
            self.leftWrist_gt.append(confidence_gt[5])
            self.rightWrist_gt.append(confidence_gt[6])
            self.leftHip_gt.append(confidence_gt[7])
            self.rightHip_gt.append(confidence_gt[8])

            self.head_pred.append(confidence_pred[0])
            self.leftShoulder_pred.append(confidence_pred[1])
            self.rightShoulder_pred.append(confidence_pred[2])
            self.leftElbow_pred.append(confidence_pred[3])
            self.rightElbow_pred.append(confidence_pred[4])
            self.leftWrist_pred.append(confidence_pred[5])
            self.rightWrist_pred.append(confidence_pred[6])
            self.leftHip_pred.append(confidence_pred[7])
            self.rightHip_pred.append(confidence_pred[8])

    def auc_cal_all(self):
        try:
            auc_all = roc_auc_score(self.confidences_gt_all,self.confidences_pred_all)
        except ValueError:
            auc_all = 0

        return auc_all

    def auc_cal(self):
        auc_head = roc_auc_score(self.head_gt,self.head_pred) if 0 in self.head_gt else 0
        auc_leftShoulder = roc_auc_score(self.leftShoulder_gt,self.leftShoulder_pred) if 0 in self.leftShoulder_gt else 0
        auc_rightShoulder = roc_auc_score(self.rightShoulder_gt,self.rightShoulder_pred) if 0 in self.rightShoulder_gt else 0
        auc_leftElbow = roc_auc_score(self.leftElbow_gt,self.leftElbow_pred) if 0 in self.leftElbow_gt else 0
        auc_rightElbow = roc_auc_score(self.rightElbow_gt,self.rightElbow_pred) if 0 in self.rightElbow_gt else 0
        auc_leftWrist = roc_auc_score(self.leftWrist_gt,self.leftWrist_pred) if 0 in self.leftWrist_gt else 0
        auc_rightWrist = roc_auc_score(self.rightWrist_gt,self.rightWrist_pred) if 0 in self.rightWrist_gt else 0
        auc_leftHip = roc_auc_score(self.leftHip_gt,self.leftHip_pred) if 0 in self.leftHip_gt else 0
        auc_rightHip = roc_auc_score(self.rightHip_gt,self.rightHip_pred) if 0 in self.rightHip_gt else 0
        auc_leftKnee = roc_auc_score(self.leftKnee_gt,self.leftKnee_pred) if 0 in self.leftKnee_gt else 0
        auc_rightKnee = roc_auc_score(self.rightKnee_gt,self.rightKnee_pred) if 0 in self.rightKnee_gt else 0
        auc_leftAnkle = roc_auc_score(self.leftAnkle_gt,self.leftAnkle_pred) if 0 in self.leftAnkle_gt else 0
        auc_rightAnkle = roc_auc_score(self.rightAnkle_gt,self.rightAnkle_pred) if 0 in self.rightAnkle_gt else 0

        return auc_head,auc_leftShoulder,auc_rightShoulder,auc_leftElbow,auc_rightElbow,auc_leftWrist,\
               auc_rightWrist,auc_leftHip,auc_rightHip,auc_leftKnee,auc_rightKnee,auc_leftAnkle,auc_rightAnkle

