from statistics import mean
from opt import opt
import numpy as np
from src.utils.pose import Pose2D

class pose_eval:
    def __init__(self):
        self.distThresh = 0.2
        self.distances = []
        self.pcks = []
        self.dist_kp = []
        self.PCKH = {}


    def save_value(self,pose_gt,pose_pred):

        ALL_dist, KP_dist, distances = pose_gt.distance_to(pose_pred)
        match = ALL_dist <= self.distThresh
        pck = 1.0 * np.sum(match, axis=0) / len(ALL_dist)
        self.pcks.append(pck)
        self.dist_kp.append(KP_dist)
        self.distances.append(distances)
        return self.pcks, self.dist_kp, self.distances


    def cal_eval(self):

        kps_acc = self.cal_kps_acc(self.dist_kp)

        return kps_acc


    def cal_kps_acc(self,kp_acc):
        value = []
        acc = []
        j = 0
        for i in range(len(kp_acc[j])):
            for item in kp_acc:
                value.append([item[i][0]])
            match_KP = np.array(value) <= self.distThresh
            pck_KP = 1.0 * np.sum(match_KP, axis=0) / len(value)
            acc.append(pck_KP)
            j = j + 1
        return acc

    def cal_pckh(self, y_pred, y_true, if_exist, thre, joints):
        a = joints
        parts_valid = sum(if_exist)[-a:].tolist()
        parts_correct, pckh = [0] * a, []
        for i in range(len(y_true)):
            if joints == 16:
                central = (y_true[i][-3] + y_true[i][-4]) / 2
                head_size = 2 * np.linalg.norm(np.subtract(central, y_true[i][-7]))
            else:
                central = (y_true[i][1] + y_true[i][2]) / 2
                head_size = 2 * np.linalg.norm(np.subtract(central, y_true[i][0]))
            if head_size == 0:
                head_size = 1e-6
            valid = np.array(if_exist[i][-a:])
            dist = np.linalg.norm(y_true[i][-a:] - y_pred[i][-a:], axis=1)
            ratio = dist / head_size
            scale = ratio * valid
            correct_num = sum((0 < scale) & (scale <= thre))  # valid_joints(a)
            pckh.append(correct_num / sum(valid)) if sum(valid) > 0 else pckh.append(0)

            for idx, (s, v) in enumerate(zip(scale, valid)):
                if v == 1 and s <= thre:
                    parts_correct[idx] += 1

        self.parts_pckh = []
        for correct_pt, valid_pt in zip(parts_correct, parts_valid):
            self.parts_pckh.append(correct_pt / valid_pt) if valid_pt > 0 else self.parts_pckh.append(0)

        return self.parts_pckh + [sum(pckh) / len(pckh)]
