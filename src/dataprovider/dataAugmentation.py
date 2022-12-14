from src.utils.pose import Pose2D, PoseConfig
import random
import cv2
import numpy as np
from opt import opt

class DataAugmentation:


    def __init__(self):

        if opt.totaljoints == 13:
            self.sym_permutation = [i for i in range(len(PoseConfig.NAMES))]
            self.sym_permutation[PoseConfig.L_SHOULDER] = PoseConfig.R_SHOULDER
            self.sym_permutation[PoseConfig.R_SHOULDER] = PoseConfig.L_SHOULDER
            self.sym_permutation[PoseConfig.L_ELBOW] = PoseConfig.R_ELBOW
            self.sym_permutation[PoseConfig.R_ELBOW] = PoseConfig.L_ELBOW
            self.sym_permutation[PoseConfig.L_WRIST] = PoseConfig.R_WRIST
            self.sym_permutation[PoseConfig.R_WRIST] = PoseConfig.L_WRIST
            self.sym_permutation[PoseConfig.L_HIP] = PoseConfig.R_HIP
            self.sym_permutation[PoseConfig.R_HIP] = PoseConfig.L_HIP
            self.sym_permutation[PoseConfig.R_KNEE] = PoseConfig.L_KNEE
            self.sym_permutation[PoseConfig.L_KNEE] = PoseConfig.R_KNEE
            self.sym_permutation[PoseConfig.L_ANKLE] = PoseConfig.R_ANKLE
            self.sym_permutation[PoseConfig.R_ANKLE] = PoseConfig.L_ANKLE
        elif opt.totaljoints == 9:
            self.sym_permutation = [i for i in range(len(PoseConfig.RENAMES))]
            self.sym_permutation[PoseConfig.L_SHOULDER] = PoseConfig.R_SHOULDER
            self.sym_permutation[PoseConfig.R_SHOULDER] = PoseConfig.L_SHOULDER
            self.sym_permutation[PoseConfig.L_ELBOW] = PoseConfig.R_ELBOW
            self.sym_permutation[PoseConfig.R_ELBOW] = PoseConfig.L_ELBOW
            self.sym_permutation[PoseConfig.L_WRIST] = PoseConfig.R_WRIST
            self.sym_permutation[PoseConfig.R_WRIST] = PoseConfig.L_WRIST
            self.sym_permutation[PoseConfig.L_HIP] = PoseConfig.R_HIP
            self.sym_permutation[PoseConfig.R_HIP] = PoseConfig.L_HIP
        elif opt.dataset == "MPII":
            self.sym_permutation = [i for i in range(len(PoseConfig.MPIINAMES))]
            self.sym_permutation[PoseConfig.MPIIl_shoulder] = PoseConfig.MPIIr_shoulder
            self.sym_permutation[PoseConfig.MPIIr_shoulder] = PoseConfig.MPIIl_shoulder
            self.sym_permutation[PoseConfig.MPIIl_elbow] = PoseConfig.MPIIr_elbow
            self.sym_permutation[PoseConfig.MPIIr_elbow] = PoseConfig.MPIIl_elbow
            self.sym_permutation[PoseConfig.MPIIl_wrist] = PoseConfig.MPIIr_wrist
            self.sym_permutation[PoseConfig.MPIIr_wrist] = PoseConfig.MPIIl_wrist
            self.sym_permutation[PoseConfig.MPIIl_hip] = PoseConfig.MPIIr_hip
            self.sym_permutation[PoseConfig.MPIIr_hip] = PoseConfig.MPIIl_hip
            self.sym_permutation[PoseConfig.MPIIr_knee] = PoseConfig.MPIIl_knee
            self.sym_permutation[PoseConfig.MPIIl_knee] = PoseConfig.MPIIr_knee
            self.sym_permutation[PoseConfig.MPIIl_ankle] = PoseConfig.MPIIr_ankle
            self.sym_permutation[PoseConfig.MPIIr_ankle] = PoseConfig.MPIIl_ankle
        else:
            raise ValueError("Your dataset name is wrong")


    def apply(self, image,poses):

        image = self._distort_image(image)

        if (random.random() > 0.5):
            image, poses = self._symetry(image,poses)

        return image, poses
    def random_subsample(self, image):

        if random.random() < 0.20:

            size_reduction = 0.5 + 0.5*random.random()

            initWidth = image.shape[1]
            initHeight = image.shape[0]

            width = int(image.shape[1]*size_reduction)
            height = int(image.shape[0]*size_reduction)

            image = cv2.resize(image, (width, height))
            image = cv2.resize(image, (initWidth, initHeight))

        return image

    # def _rand_scale(self, s):
    #     scale = random.uniform(1, s)
    #     if (random.randint(1, 10000) % 2):
    #         return scale
    #     return 1. / scale


    def _distort_image(self, image):
        try:
            image = image.astype(np.float32)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

            if random.random() < 0.50:
                if random.random() < 0.50:
                    image[:, :, 2] = image[:, :, 2] * (0.4 + 0.40 * random.random())
                else:
                    image[:, :, 2] = image[:, :, 2] * (0.4 + 0.60 * random.random())

            image = np.clip(image, 0, 255)
            image = image.astype(np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
            image = self.random_subsample(image)
        except TypeError:
            pass
        return image

    def _symetry(self, image,poses):
        image = cv2.flip(image, 1)
        new_poses = []

        for pose in poses:
            joints = pose.get_joints()
            is_active_joints = pose.get_active_joints()
            joints[is_active_joints, 0] = 1.0 - joints[is_active_joints, 0]
            joints = joints[self.sym_permutation, :]
            new_poses.append(Pose2D(joints))

        return image, new_poses

