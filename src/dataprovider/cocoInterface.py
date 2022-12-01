import json
from src.utils.pose import Pose2D
import numpy as np
import os
import matplotlib.image as mpimg


class CocoInterface:

    def __init__(self, image_dir, images, annotations,datatype):
        self.images = images
        self.annotations = annotations
        self.img_dir = image_dir
        self.annotations_keys = [imgId for imgId in self.annotations.keys()]
        self.datatype = datatype

    def build(annot_file, image_dir,datatype):
        annot = json.load(open(annot_file, 'r'))

        if datatype == 'coco' or datatype == 'coco_mpii' or datatype == 'coco_mpii_13' or \
                datatype == 'coco_crowd' or datatype == 'coco_ochuman' or datatype == 'coco_human36' or \
                datatype == "ai_coco":
            images_res = {}
            # if annot_file[:-5]=='img/ai_challenger/aic_train' or annot_file[:-5]=='img/ai_challenger/aic_val':
            #     for i in range(len(annot['images'])):
            #         images_res[annot['images'][i]['id']] = {
            #             'fileName': annot_file[:-5] +'/'+annot['images'][i]['file_name'],
            #             'width': annot['images'][i]['width'],
            #             'height': annot['images'][i]['height']
            #         }
            # else:
            for i in range(len(annot['images'])):
                images_res[annot['images'][i]['id']] = {
                    'fileName': annot['images'][i]['file_name'],
                    'width': annot['images'][i]['width'],
                    'height': annot['images'][i]['height']
                }

            # build imgId => annotations
            annotations_res = {}

            for i in range(len(annot['annotations'])):
                entry = annot['annotations'][i]
                img_id = entry['image_id']
                kp = entry['keypoints']
                img_width, img_height = float(images_res[img_id]['width']), float(images_res[img_id]['height'])
                if datatype == 'coco_mpii' or datatype == 'coco_mpii_13':
                    kp = [(x / img_width, y / img_height) if v >= 1 else (-1, -1) for x, y, v in
                          zip(kp[0::3], kp[1::3], kp[2::3])]
                else:
                    kp = [(kp[i] / img_width, kp[i + 1] / img_height) for i in range(len(kp))[::2]]
                kp = np.array(kp).astype(np.float32)
                pose = Pose2D.build_from_coco(kp, datatype)

                if not img_id in annotations_res:
                    annotations_res[img_id] = []

                annotations_res[img_id].append({
                    'pose': pose
                })

        elif datatype == 'yoga' or datatype == 'rehab':
            images_res = {}
            for i in range(len(annot['images'])):
                images_res[annot['images'][i]['file_name']] = {
                    'fileName': annot['images'][i]['file_name'],
                    'width': annot['images'][i]['width'],
                    'height': annot['images'][i]['height']
                }

            annotation_res = {}
            annotations_res = {}

            for i in range(len(annot['annotations'])):
                annotation_res[annot['annotations'][i]['image_id']] = {
                    'image_id': annot['annotations'][i]['image_id'],
                    'keypoints': annot['annotations'][i]['keypoints']}
                entry = annot['annotations'][i]
                img_id = entry['image_id']
                kp = entry['keypoints']

                img_width, img_height = float(images_res[img_id]['width']), float(images_res[img_id]['height'])
                kp = [(x / img_width, y / img_height) if v >= 1 else (-1, -1) for x, y, v in
                      zip(kp[0::3], kp[1::3], kp[2::3])]
                kp = np.array(kp).astype(np.float32)
                pose = Pose2D.build_from_coco(kp, datatype)
                if not img_id in annotations_res:
                    annotations_res[img_id] = []

                annotations_res[img_id].append({
                    'pose': pose,

                })
            images_res = CocoInterface.check(images_res, annotation_res)
        return CocoInterface(image_dir, images_res, annotations_res,datatype)

    def check(images,annotations):
        images_new = {}
        for index in images:
            if index in annotations:
                images_new[index] = images[index]
            else:
                continue
        return images_new


    def size(self):
        return len(self.annotations_keys)

    def get_image(self, entry_id):
        if self.datatype == 'coco' or self.datatype == 'coco_mpii' or self.datatype == 'coco_mpii_13' or \
                self.datatype == 'coco_crowd' or self.datatype == 'coco_ochuman' or self.datatype == 'coco_human36':

            # img_path = os.path.join(self.img_dir, self.images[self.annotations_keys[entry_id]]["fileName"])
            img_path = os.path.join(self.images[self.annotations_keys[entry_id]]["fileName"])
        elif self.datatype == 'yoga' or self.datatype == 'rehab':
            img_path = os.path.join(self.img_dir,self.images[self.annotations_keys[entry_id]]["fileName"])

        return mpimg.imread(img_path)

    def get_image_shape(self, entry_id):
        width = self.images[self.annotations_keys[entry_id]]['width']
        height = self.images[self.annotations_keys[entry_id]]['height']
        return [height, width, 3]

    def get_total_person_on(self, entry_id):
        return len(self.annotations[self.annotations_keys[entry_id]])

    def get_pose(self, entry_id, person_id):
        return self.annotations[self.annotations_keys[entry_id]][person_id]['pose']

    def get_poses(self, entry_id):
        return [self.get_pose(entry_id, person_id) for person_id in range(self.get_total_person_on(entry_id))]
