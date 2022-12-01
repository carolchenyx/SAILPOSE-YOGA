from opt import opt

cmds = [
    # lr_type
    "CUDA_VISIBLE_DEVICES=0 python train.py --batch 8 --epoch 1 --fromStep 0 --Early_stopping=True --isTrain=True --isTrainpre=True --modeloutputFile Yogapose --checkpoinsaveDir Yogapose --train_all_result Result/Yogapose --backbone mobilenetv2 --lr_type exponential_decay --lr 0.0030000 --decay_rate 0.98 --decay_steps 3000 --optimizer Gradient --activate_function relu --Model_folder_name 0",
    "CUDA_VISIBLE_DEVICES=0 python train.py --batch 8 --epoch 1 --fromStep 0 --Early_stopping=True --isTrain=True --isTrainpre=True --modeloutputFile Yogapose --checkpoinsaveDir Yogapose --train_all_result Result/Yogapose --backbone mobilenetv2 --lr_type polynomial_decay --lr 0.0030000 --decay_rate 0.98 --decay_steps 3000 --optimizer Gradient --activate_function relu --Model_folder_name 1",
    "CUDA_VISIBLE_DEVICES=0 python train.py --batch 8 --epoch 1 --fromStep 0 --Early_stopping=True --isTrain=True --isTrainpre=True --modeloutputFile Yogapose --checkpoinsaveDir Yogapose --train_all_result Result/Yogapose --backbone mobilenetv2 --lr_type inverse_time_decay --lr 0.0030000 --decay_rate 0.98 --decay_steps 3000 --optimizer Gradient --activate_function relu --Model_folder_name 2",
    "CUDA_VISIBLE_DEVICES=0 python train.py --batch 8 --epoch 1 --fromStep 0 --Early_stopping=True --isTrain=True --isTrainpre=True --modeloutputFile Yogapose --checkpoinsaveDir Yogapose --train_all_result Result/Yogapose --backbone mobilenetv2 --lr_type cosine_decay --lr 0.0030000 --decay_rate 0.98 --decay_steps 3000 --optimizer Gradient --activate_function relu --Model_folder_name 3",

    # lr

    # decay_rate

    # decay_steps
]

import os
os.makedirs("Result/{}".format(opt.modeloutputFile), exist_ok=True)
if not os.path.exists("Result/Yogapose/train_log.log"):
    os.mknod("Result/Yogapose/train_log.log")
log = open("Result/Yogapose/train_log.log", "a+")
log.write("\n")
log.write("try different lr_type")
log.write("\n")
for cmd in cmds:
    log.write(cmd)
    log.write("\n")
    os.system(cmd)