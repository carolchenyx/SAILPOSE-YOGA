
import tensorflow as tf
import math



def wing_loss(predHeat, gtHeat, w=10.0, epsilon=2.0):
    with tf.name_scope('wing_loss'):
        x = predHeat - gtHeat
        c = w * (1.0 - math.log(1.0 + w / epsilon))
        absolute_x = tf.abs(x)
        losses = tf.where(
            tf.greater(w, absolute_x),
            w * tf.log(1.0 + absolute_x / epsilon),
            absolute_x - c
        )
        loss = tf.reduce_mean(tf.reduce_sum(losses))
        tf.cast(loss, tf.float32)
        return loss



def adaptivewingLoss(predHeat, gtHeat, omega=14.0, theta=0.5, epsilon=1.0, alpha=2.1):
    delta_y = predHeat-gtHeat
    delta_y = tf.abs(delta_y)
    delta_y1 = delta_y[delta_y < theta]
    a = tf.div(delta_y1,omega)
    b = tf.div(epsilon,theta)
    delta_y2 = delta_y[delta_y >= theta]
    y1 = delta_y[delta_y < theta]
    y2 = delta_y[delta_y >= theta]
    # y[delta_y >= theta]
    loss1 = omega * tf.log(1 + tf.pow(a, alpha - y1))
    A = omega * (1 / (1 + tf.pow(b, alpha - y2))) * (alpha - y2) * (tf.pow(b, alpha - y2 - 1)) * (1 / epsilon)
    C = theta * A - omega * tf.log(1 + tf.pow(b, alpha - y2))
    loss2 = A * delta_y2 - C

    data_len = tf.size(loss1) + tf.size(loss2)
    data_len = tf.cast(data_len,tf.float32)
    return tf.div((tf.reduce_sum(loss1)+tf.reduce_sum(loss2)),data_len)
    # return (loss1.sum() + loss2.sum()) /(len(loss1) + len(loss2))

def smooth_l1_loss(self, predHeat, gtHeat, scope=tf.GraphKeys.LOSSES):
    with tf.variable_scope(scope):
        absolute_x = tf.abs(predHeat - gtHeat)
        sq_x = 0.5*absolute_x**2
        less_one = tf.cast(tf.less(absolute_x, 1.0), tf.float32)
        smooth_l1_loss = (less_one*sq_x)+(1.0-less_one)
        return smooth_l1_loss


