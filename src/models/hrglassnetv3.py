import tensorflow as tf
import tensorflow.contrib.slim as slim
from opt import opt
from src.models.network_base import max_pool, upsample, inverted_bottleneck, separable_conv, convb, is_trainable

out_channel_ratio = lambda d: int(d * 1.0)
up_channel_ratio = lambda d: int(d * 1.0)

l2s = []


class HourGlassNet:
    def __init__(self, is4Train=True, stageNum=4, totalJoints=opt.totaljoints):
        is_trainable(is4Train)
        self.input = tf.placeholder(tf.float32, shape=(None, 256, 256, 3), name='Image')
        net = convb(self.input, 3, 3, out_channel_ratio(16), 2, name="Conv2d_0")

        # 128, 112
        net = slim.stack(net, inverted_bottleneck,
                         [
                             (1, out_channel_ratio(16), 0, 3),
                             (1, out_channel_ratio(16), 0, 3)
                         ], scope="Conv2d_1")

        # 64, 56
        net = slim.stack(net, inverted_bottleneck,
                         [
                             (up_channel_ratio(6), out_channel_ratio(24), 1, 3),
                             (up_channel_ratio(6), out_channel_ratio(24), 0, 3),
                             (up_channel_ratio(6), out_channel_ratio(24), 0, 3),
                             (up_channel_ratio(6), out_channel_ratio(24), 0, 3),
                             (up_channel_ratio(6), out_channel_ratio(24), 0, 3),
                         ], scope="Conv2d_2")

        net_h_w = int(net.shape[1])
        # build network recursively
        hg_out = LayerProvider.hourglass_module(net, stageNum, totalJoints)

        for index, l2 in enumerate(l2s):
            l2_w_h = int(l2.shape[1])
            if l2_w_h == net_h_w:
                continue
            scale = net_h_w // l2_w_h
            l2s[index] = upsample(l2, scale, name="upsample_for_loss_%d" % index)

        self.output = hg_out
        self.intermediate_out = l2s

    def getInput(self):
        return self.input

    def getOutput(self):
        return self.output

    def getInterOut(self):
        return self.intermediate_out

class LayerProvider:
    def hourglass_module(inp, stageNum, totalJoints):
        if stageNum > 0:
            down_sample = max_pool(inp, 2, 2, 2, 2, name="hourglass_downsample_%d" % stageNum)
    
            block_front = slim.stack(down_sample, inverted_bottleneck,
                                     [
                                         (up_channel_ratio(6), out_channel_ratio(24), 0, 3),
                                         (up_channel_ratio(6), out_channel_ratio(24), 0, 3),
                                         (up_channel_ratio(6), out_channel_ratio(24), 0, 3),
                                         (up_channel_ratio(6), out_channel_ratio(24), 0, 3),
                                         (up_channel_ratio(6), out_channel_ratio(24), 0, 3),
                                     ], scope="hourglass_front_%d" % stageNum)
            stageNum -= 1
            block_mid = LayerProvider.hourglass_module(block_front, stageNum, totalJoints)
            block_back = inverted_bottleneck(
                block_mid, up_channel_ratio(6), totalJoints,
                0, 3, scope="hourglass_back_%d" % stageNum)
    
            up_sample = upsample(block_back, 2, "hourglass_upsample_%d" % stageNum)
    
            # jump layer
            branch_jump = slim.stack(inp, inverted_bottleneck,
                                     [
                                         (up_channel_ratio(6), out_channel_ratio(24), 0, 3),
                                         (up_channel_ratio(6), out_channel_ratio(24), 0, 3),
                                         (up_channel_ratio(6), out_channel_ratio(24), 0, 3),
                                         (up_channel_ratio(6), out_channel_ratio(24), 0, 3),
                                         (up_channel_ratio(6), totalJoints, 0, 3),
                                     ], scope="hourglass_branch_jump_%d" % stageNum)
    
            curr_hg_out = tf.add(up_sample, branch_jump, name="hourglass_out_%d" % stageNum)
            # mid supervise
            l2s.append(curr_hg_out)
    
            return curr_hg_out
    
        _ = inverted_bottleneck(
            inp, up_channel_ratio(6), out_channel_ratio(24),
            0, 3, scope="hourglass_mid_%d" % stageNum
        )
        return _


