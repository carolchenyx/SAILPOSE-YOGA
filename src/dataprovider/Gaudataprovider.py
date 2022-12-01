from src.dataprovider.dataAdaptator import DataAdaptator
from src.utils.body_cover import BodyCover
from src.dataprovider.dataAugmentation import DataAugmentation
import random
import numpy as np
from src.dataprovider.cocoInterface import CocoInterface
from src.utils.pose import Pose2D
import matplotlib.pyplot as plt
from src.utils.drawer import Drawer
import math
from opt import opt
from Config import config_cmd as config


class GauDataProviderAdaptator:

    def __init__(self, annotFile, imageDir, inputSize, outputSize, batchSize, datatype, offset=opt.offset):

        self.provider = DataProvider.build(annotFile, imageDir, inputSize, batchSize, datatype)
        self.provider_vis = DataProvider.build_vis(annotFile, imageDir, inputSize, batchSize, datatype)
        self.outputSize = outputSize
        self.inputSize = inputSize
        self.batchSize = batchSize
        self.th = opt.gaussian_thres
        self.sigma = opt.gaussian_sigma
        self.offset = offset
        self.jonit_weight = config.joint_weight
        self.use_different_joint_weights = config.use_different_joint_weights
        if opt.checkanno == True:
            self.debug()

    def put_heatmap(self, heatmap, plane_idx, center, sigma):
        center_x, center_y = center
        # _, height, width = heatmap.shape[:3]
        height = self.outputSize[0]
        width = self.outputSize[1]
        delta = math.sqrt(self.th * 2)

        x0 = int(max(0, center_x - delta * sigma))
        y0 = int(max(0, center_y - delta * sigma))

        x1 = int(min(width, center_x + delta * sigma))
        y1 = int(min(height, center_y + delta * sigma))

        # gaussian filter
        for y in range(y0, y1):
            for x in range(x0, x1):
                d = (x - center_x) ** 2 + (y - center_y) ** 2
                exp = d / 2.0 / sigma / sigma
                if exp > self.th:
                    continue
                # heatmap[plane_idx][y][x] = max(heatmap[plane_idx][y][x], math.exp(-exp))
                # heatmap[plane_idx][y][x] = min(heatmap[plane_idx][y][x], 1.0)
                # Format = [0, y, x, 0]
                heatmap[plane_idx][0, y, x, 0] = max(heatmap[plane_idx][0, y, x, 0], math.exp(-exp))
                heatmap[plane_idx][0, y, x, 0] = min(heatmap[plane_idx][0, y, x, 0], 1.0)

    def decodePoses(self, outputs):
        poses = []
        totalJoints = opt.totaljoints
        for recordId in range(outputs.shape[0]):
            if self.offset == True:
                heatmap = outputs[recordId, :, :, :totalJoints]
                xOff = outputs[recordId, :, :, totalJoints:(totalJoints * 2)]
                yOff = outputs[recordId, :, :, (totalJoints * 2):]
            else:
                heatmap = outputs[recordId, :, :, :totalJoints]
            joints = np.zeros((totalJoints, 2)) - 1
            for jointId in range(totalJoints):
                inlinedPix = heatmap[:, :, jointId].reshape(-1)
                pixId = np.argmax(inlinedPix)
                # if max confidence below 0.1 => inactive joint
                if inlinedPix[pixId] < 0.01:
                    continue
                outX = pixId % self.outputSize[0]
                outY = pixId // self.outputSize[0]
                if self.offset == True:
                    x = outX / self.outputSize[0] * self.inputSize[0] + xOff[outY, outX, jointId]
                    y = outY / self.outputSize[1] * self.inputSize[1] + yOff[outY, outX, jointId]
                else:
                    x = outX / self.outputSize[0] * self.inputSize[0]
                    y = outY / self.outputSize[1] * self.inputSize[1]
                x = x / self.inputSize[0]
                y = y / self.inputSize[1]
                joints[jointId, 0] = x
                joints[jointId, 1] = y
            poses.append(Pose2D(joints))
        return poses

    def processAnnotationsnooffset(self, outputs):

        annotations = []
        #count = 0
        for pose in outputs:

            joints = pose.get_joints()

            # clamp below one so that : id = xValue*imgSize is consistant
            joints = np.minimum(joints, 0.99999)

            heatmaps = []

            for jointId in range(opt.totaljoints):
                center=[]

                x, y = int(joints[jointId, 0] * self.outputSize[0]), int(joints[jointId, 1] * self.outputSize[1])
                center.append(x)
                center.append(y)
                tmp = np.zeros((1, self.outputSize[1], self.outputSize[0], 1))
                if pose.is_active_joint(jointId):
                    tmp[0, y, x, 0] = 1.0

                heatmaps.append(tmp)
                sigma = self.sigma # This will control the size of the gaussian filter, I guess this shld be altered according to size of HM
                self.put_heatmap(heatmaps,jointId,center,sigma)


            heatmaps = np.concatenate(heatmaps, 3)
            annot = heatmaps
            annotations.append(annot)

        annotations = np.concatenate(annotations, 0)

        return annotations

    def processAnnotationswithoffset(self, outputs):

        annotations = []
        if_exit = []
        for pose in outputs:

            joints = pose.get_joints()
            if_exit.append(pose.get_active_joints())
            # clamp below one so that : id = xValue*imgSize is consistant
            joints = np.minimum(joints, 0.99999)

            heatmaps, xOffsets, yOffsets = [], [], []

            for jointId in range(opt.totaljoints):
                center = []
                x, y = int(joints[jointId, 0] * self.outputSize[0]), int(joints[jointId, 1] * self.outputSize[1])
                center.append(x)
                center.append(y)
                tmp = np.zeros((1, self.outputSize[1], self.outputSize[0], 1))
                if pose.is_active_joint(jointId):
                    tmp[0, y, x, 0] = 1.0

                heatmaps.append(tmp)
                sigma = self.sigma  # This will control the size of the gaussian filter, I guess this shld be altered according to size of HM
                self.put_heatmap(heatmaps, jointId, center, sigma)

                offX, offY = int(joints[jointId, 0] * self.inputSize[0]), int(joints[jointId, 1] * self.inputSize[1])
                offX, offY = offX - int((x / self.outputSize[0]) * self.inputSize[0]), offY - int(
                    (y / self.outputSize[1]) * self.inputSize[1])

                tmp = np.zeros((1, self.outputSize[1], self.outputSize[0], 1))
                if pose.is_active_joint(jointId):
                    tmp[0, y, x, 0] = offX
                xOffsets.append(tmp)

                tmp = np.zeros((1, self.outputSize[1], self.outputSize[0], 1))
                if pose.is_active_joint(jointId):
                    tmp[0, y, x, 0] = offY
                yOffsets.append(tmp)

            heatmaps = np.concatenate(heatmaps, 3)
            xOffsets = np.concatenate(xOffsets, 3)
            yOffsets = np.concatenate(yOffsets, 3)
            annot = np.concatenate([heatmaps, xOffsets, yOffsets], 3)

            annotations.append(annot)

        annotations = np.concatenate(annotations, 0)

        return annotations, if_exit

    def debug(self):

        # get inputs images + poses annot
        inputsOrig, outputsOrig = self.provider.drawn()
        if self.offset == True:
            # build encoded annotation
            inputs, outputs = inputsOrig, self.processAnnotationswithoffset(outputsOrig)
        else:
            inputs, outputs = inputsOrig, self.processAnnotationsnooffset(outputsOrig)

        # decode the annotation back to a pose class
        poses = self.decodePoses(outputs[0])

        for recordId in range(inputs.shape[0]):
            for i in range(opt.totaljoints):
                isOk = outputs[0][recordId, :, :, i].sum() == 1

            img = (((inputs[recordId, :, :, :] + 1) / 2) * 255).astype(np.uint8)

            print("encoded-decoded")
            img1 = Drawer().draw_2d_pose(img, poses[recordId])
            plt.imshow(img1)
            plt.show()

            print("original")
            img2 = Drawer().draw_2d_pose(img, outputsOrig[recordId])
            plt.imshow(img2)
            plt.show()

    def getBatchSize(self):
        return self.batchSize

    def getjoint_weight(self):
        return self.jonit_weight

    def getuse_different_joint_weights(self):
        return self.use_different_joint_weights

    def drawn(self):
        inputs, outputs = self.provider.drawn()
        if self.offset == True:
            return inputs, self.processAnnotationswithoffset(outputs)
        else:
            return inputs, self.processAnnotationsnooffset(outputs)

    def drawn_visu(self):
        inputs, outputs = self.provider_vis.drawn_visu()
        if self.offset == True:
            return inputs, self.processAnnotationswithoffset(outputs)
        else:
            return inputs, self.processAnnotationsnooffset(outputs)

    def drawn_test(self):
        inputs, outputs = self.provider_vis.drawn_test()
        if self.offset == True:
            return inputs, self.processAnnotationswithoffset(outputs)
        else:
            return inputs, self.processAnnotationsnooffset(outputs)

    def get_data_leng(self):
        return self.provider.get_date_len()


class DataProvider:

    def __init__(self, coco, input_size, batch_size, padding, jitter, mask, body_cover=None, data_augment=None):

        self.adapted_datas = []

        self.batch_size = batch_size

        for img_id in range(coco.size()):
            curr_mask = mask[img_id] if not isinstance(mask, type(None)) else None
            ada_data = DataAdaptator(coco, img_id, input_size, padding, jitter, curr_mask, body_cover, data_augment)
            self.adapted_datas.append(ada_data)

        self.adapted_datas = [d for d in self.adapted_datas if d.size() > 0]
        self.image_id = 0
        total_examples = 0
        for i in range(len(self.adapted_datas)):
            total_examples += self.adapted_datas[i].size()
        print("TOTAL ANNOTATED EXAMPLES : " + str(total_examples))
        self.tmpint = random.sample(range(0, len(self.adapted_datas)), len(self.adapted_datas))

    def size(self):
        return len(self.adapted_datas)

    def get_image(self, img_id):
        return self.adapted_datas[img_id].get_image()

    def total_poses_on(self, img_id):
        return self.adapted_datas[img_id].size()

    def get_pose(self, img_id, entry_id):
        return self.adapted_datas[img_id].get_pose(entry_id)

    def ramdom_data(self):
        self.tmpint = random.sample(range(0, len(self.adapted_datas)), len(self.adapted_datas))

    def get_data(self, leng):
        if self.image_id < leng:
            id = self.tmpint[self.image_id]
            self.image_id += 1
            return id
        else:
            self.ramdom_data()
            self.image_id = 0
            id = self.tmpint[self.image_id]
            return id

    def get_date_len(self):
        return len(self.adapted_datas)

    def drawn(self):
        inputs, outputs = [], []
        for i in range(self.batch_size):
            rnd_img_id = self.get_data(len(self.adapted_datas))
            curr_in, curr_out = self.adapted_datas[rnd_img_id].drawn()
            inputs.append(np.expand_dims(curr_in, 0))
            outputs.append(curr_out)
        return np.concatenate(inputs, 0), outputs

    def drawn_test(self, batch=opt.testbatch):
        inputs, outputs = [], []
        for i in range(batch):
            rnd_img_id = self.get_data(len(self.adapted_datas))
            curr_in, curr_out = self.adapted_datas[rnd_img_id].drawn()
            if curr_in.shape[2] != 3:
                 curr_in = curr_in[:,:,:3]
            inputs.append(np.expand_dims(curr_in, 0))
            outputs.append(curr_out)
        return np.concatenate(inputs, 0), outputs

    def drawn_visu(self):

        inputs, outputs = [], []

        for i in range(self.batch_size):
            rnd_img_id = int(0.1 * len(self.adapted_datas))
            curr_in, curr_out = self.adapted_datas[rnd_img_id].drawn_vis()
            inputs.append(np.expand_dims(curr_in, 0))
            outputs.append(curr_out)

        return np.concatenate(inputs, 0), outputs

    @staticmethod
    def build(cocoAnnotFile, cocoImgDir, inputSize, batchSize, datatype):

        coco = CocoInterface.build(cocoAnnotFile, cocoImgDir, datatype)

        mask = DataProvider.active_pose_mask(coco)

        body_cover = BodyCover(0.3)

        padding, jitter = 0.4, 0.3

        data_augment = DataAugmentation()

        return DataProvider(coco, inputSize, batchSize, padding, jitter, mask=mask, body_cover=body_cover,
                            data_augment=data_augment)

    def build_vis(cocoAnnotFile, cocoImgDir, inputSize, batchSize, datatype):

        coco = CocoInterface.build(cocoAnnotFile, cocoImgDir, datatype)

        mask = DataProvider.active_pose_mask(coco)

        body_cover = BodyCover(0.3)

        padding, jitter = 0.4, 0.3

        data_augment = DataAugmentation()

        return DataProvider(coco, inputSize, batchSize, padding, jitter, mask=mask, body_cover=body_cover,
                            data_augment=None)

    @staticmethod
    def active_pose_mask(coco):

        # at least k labeled joints
        min_annots_per_person = 8
        # at least subject bbox width or height > n_pix
        min_subject_size = 50
        # max overlapped IOU person :
        max_overlaps = 0.1

        mask = []
        for imgId in range(coco.size()):
            curr_mask = []
            for personId in range(coco.get_total_person_on(imgId)):
                is_annotation_selected = True

                # at least k labeled joints
                is_annotation_selected &= coco.get_pose(imgId, personId).total_labeled_joints() >= min_annots_per_person

                # at least subject bbox width or height > n_pix
                subject_bbox = coco.get_pose(imgId, personId).to_bbox()
                img_shape = coco.get_image_shape(imgId)
                subject_width = (subject_bbox.get_max_x() - subject_bbox.get_min_x()) * img_shape[1]
                subject_height = (subject_bbox.get_max_y() - subject_bbox.get_min_y()) * img_shape[0]
                max_size = max(subject_width, subject_height)

                is_annotation_selected &= max_size > min_subject_size

                # max overlapped IOU person :

                is_overlapped_ok = True

                for another_person_id in range(coco.get_total_person_on(imgId)):

                    if another_person_id == personId:
                        continue

                # if multiple id, add in here

                is_annotation_selected &= is_overlapped_ok

                curr_mask.append(is_annotation_selected)

            mask.append(curr_mask)

        return mask
