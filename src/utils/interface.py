import tensorflow as tf
import os
import numpy as np
from src.utils.body_cover import BodyCover
from src.utils.pose import Pose2D, PoseConfig
from opt import opt

class Pose2DInterface:

    def __init__(self, session, protograph, post_processing, input_size, subject_padding, input_node_name, output_node_name,offsetornot=opt.offset):

        # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        self.offsetornot = offsetornot

        with tf.gfile.GFile(protograph, "rb") as f:
            restored_graph_def = tf.GraphDef()
            restored_graph_def.ParseFromString(f.read())

        tf.import_graph_def(
            restored_graph_def,
            input_map=None,
            return_elements=None,
            name=""
        )

        self.session = session

        self.graph = tf.get_default_graph()

        self.image = self.graph.get_tensor_by_name(input_node_name)

        self.output = self.graph.get_tensor_by_name(output_node_name)

        self.input_size = input_size

        self.post_processing = post_processing

        self.subject_padding = subject_padding

        self.body_cover = BodyCover(self.subject_padding)



    """
    In the case the model output heatmaps+offset vectors, this postprocessing transform the
    resulting output in the pose 2D. (defined for the post_processing attribute in the init method)
    """
    @staticmethod
    def our_approach_postprocessing(network_out, subject_bbox, input_size, offsetornot=opt.offset):
        total_joints = PoseConfig.get_total_joints()
        if offsetornot == True:

            heatmap = network_out[:, :, :total_joints]
            xOff = network_out[:, :, total_joints:(total_joints * 2)]
            yOff = network_out[:, :, (total_joints * 2):]

        else:
            heatmap = network_out[:, :, :total_joints]


        confidences = []
        joints = np.zeros((total_joints, 2)) - 1

        for jointId in range(total_joints):

            inlined_pix = heatmap[:, :, jointId].reshape(-1)
            pixId = np.argmax(inlined_pix)

            confidence = inlined_pix[pixId]

            # if max confidence below 0.1 => inactive joint
            if inlined_pix[pixId] < 0.01:
                confidences.append(confidence)
                continue

            outX = pixId % heatmap.shape[1]
            outY = pixId // heatmap.shape[1]

            if offsetornot ==True:
                x = outX / heatmap.shape[1] * input_size[0] + xOff[outY, outX, jointId]
                y = outY / heatmap.shape[0] * input_size[0] + yOff[outY, outX, jointId]
            else:
                x = outX / heatmap.shape[1] * input_size[0]
                y = outY / heatmap.shape[0] * input_size[0]
            x = x / input_size[0]
            y = y / input_size[0]

            joints[jointId, 0] = x
            joints[jointId, 1] = y
            confidences.append(confidence)


        return Pose2D(joints).to_absolute_coordinate_from(subject_bbox), confidences




