from opt import opt


def exponential_decay(cur_lr):
    lr = cur_lr * opt.decay_rate
    return lr

def polynomial_decay(i,cur_lr):
    global_step = min(i, 1000)
    lr = (cur_lr - 0.00001) * (1 - global_step / 1000) + 0.00001
    '''
    global_step = min(global_step, decay_steps)
    decayed_learning_rate = (learning_rate - end_learning_rate) *
                            (1 - global_step / decay_steps) ^ (power) +
                             end_learning_rate
    '''
    return lr

def inverse_time_decay(i,cur_lr):
    lr = cur_lr / (1 + opt.decay_rate * i / 1000)
    return lr
