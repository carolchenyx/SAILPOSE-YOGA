from opt import opt
from src.utils.LR import exponential_decay,polynomial_decay,inverse_time_decay


def earlystop(cur_lr,j_num,i):
        j_num += 1
        if j_num < opt.j_min:
            pass
        else:
            if opt.lr_type == "exponential_decay":
                lr = exponential_decay(cur_lr)
            elif opt.lr_type == "polynomial_decay":
                lr = polynomial_decay(i, cur_lr)
            elif opt.lr_type == "inverse_time_decay":
                lr = inverse_time_decay(i, cur_lr)
            else:
                raise ValueError("Your lr_type name is wrong")
            cur_lr = lr
        return cur_lr,j_num
