import os.path as osp
from dl_lib.configs.base_detection_config import BaseDetectionConfig

_config_dict = dict(
    MODEL=dict(
        WEIGHTS="",
        RESNETS=dict(DEPTH=18),
        PIXEL_MEAN=[0.485, 0.456, 0.406],
        PIXEL_STD=[0.229, 0.224, 0.225],
        CENTERNET=dict(
            DECONV_CHANNEL=[512, 256, 128, 64],
            DECONV_KERNEL=[4, 4, 4],
            NUM_CLASSES=1,
            MODULATE_DEFORM=True,
            BIAS_VALUE=-2.19,  # apparently this has to do with classes
            DOWN_SCALE=4,
            MIN_OVERLAP=0.7, # deleted in favor of constant rardius, I think its not needed
            TENSOR_DIM=512,   # max cap of number of instance - NOTE in 1 20x image there can be 2000 cells - lower this when we lower the image size
        ),
        LOSS=dict(
            WEIGHT_SCOREMAP=1,
            WEIGHT_OFFSET=0.0001,  # TODO for debugging some training
        ),
    ),
    INPUT=dict( # TODO this all is unnneded
        AUG=dict(
            TRAIN_PIPELINES=[
                ('CenterAffine', dict(
                    boarder=128,
                    output_size=(512, 512),
                    random_aug=True)),
                ('RandomFlip', dict()),
                ('RandomBrightness', dict(intensity_min=0.6, intensity_max=1.4)),
                ('RandomContrast', dict(intensity_min=0.6, intensity_max=1.4)),
                ('RandomSaturation', dict(intensity_min=0.6, intensity_max=1.4)),
                ('RandomLighting', dict(scale=0.1)),
            ],
            TEST_PIPELINES=[
            ],
        ),
        FORMAT="RGB",
        OUTPUT_SIZE=(128, 128),
    ),
    DATALOADER=dict(
        NUM_WORKERS=4,
    ),
    DATASETS=dict(
        TRAIN=("coco_2017_train",),
        TEST=("coco_2017_val",),
    ),
    SOLVER=dict(
        OPTIMIZER=dict(
            NAME="SGD",
            BASE_LR=0.02,
            WEIGHT_DECAY=1e-4,
        ),
        LR_SCHEDULER=dict(
            GAMMA=0.1,
            STEPS=(81000, 108000),
            MAX_ITER=126000,
            WARMUP_ITERS=1000,
        ),
        IMS_PER_BATCH=128,
        MAX_ITER=126000,  # TODO redundant with LR_SCHEDULER.MAX_ITER
    ),
    OUTPUT_DIR=osp.join(
        '/data/Outputs/model_logs/playground',
        osp.split(osp.realpath(__file__))[0].split("playground/")[-1]  # NOTE: this file was moved therefore this is bullshit currenlty
    ),
    GLOBAL=dict(DUMP_TEST=False),
    TEST=dict(
        EXPECTED_RESULTS=[],
        EVAL_PERIOD=0,
        KEYPOINT_OKS_SIGMAS=[],
        DETECTIONS_PER_IMAGE=100,  # TODO more...
        AUG=dict(  # TODO understand what is this
            ENABLED=False,
            MIN_SIZES=(400, 500, 600, 700, 800, 900, 1000, 1100, 1200),
            MAX_SIZE=4000,
            FLIP=True,
        ),
        PRECISE_BN=dict(ENABLED=False, NUM_ITER=200),  # TODO: what?
    ),
)


class CenterNetConfig(BaseDetectionConfig):
    def __init__(self):
        super(CenterNetConfig, self).__init__()
        self._register_configuration(_config_dict)


config = CenterNetConfig()
