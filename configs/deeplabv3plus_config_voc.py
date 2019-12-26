# coding=utf-8

# data
DATA = {"TYPE":"pascal",
        "TRAIN_DIR":"/home/leon/data/data/VOCdevkit/VOC2012",
        "VAL_DIR":"/home/leon/data/data/VOCdevkit/VOC2012",
        "TEST_DIR": "/home/leon/data/data/VOCdevkit/VOC2007",
        "NUM":21}

# train
TRAIN = {
         # train
         "BASE_SIZE":513,
         "CROP_SIZE":513,
         "BATCH_SIZE":4,
         "MULTI_SCALE_TRAIN":True,
         "EPOCHS":50,
         "NUMBER_WORKERS":4,
          # optim
         "MOMENTUM":0.9,
         "WEIGHT_DECAY":0.0005,
         "LR_INIT":0.007/4,
         "LR_END":1e-6,
         "WARMUP_EPOCHS":2,
         "LR_SCHEDULER":"poly",
         # loss
         "LOSS_TYPE":"ce",
         "USE_BLANCED_WEIGHTS":False,
         }

# VAL
VAL = {
        "BASE_SIZE":513,
        "CROP_SIZE":513,
        "BATCH_SIZE":4,
        "NUMBER_WORKERS":4,
        }

# TEST
TEST = {
        "BASE_SIZE":513,
        "CROP_SIZE":513,
        "BATCH_SIZE":1,
        "NUMBER_WORKERS":0,
        }