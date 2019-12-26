import utils.gpu as gpu
import torch
from torch.utils.data import DataLoader
import datasets.pascal as pascal
import time
import argparse
from evals.evaluator import Evaluator
from utils.tools import *
import configs.deeplabv3plus_config_voc as cfg
from models.nets.deeplab import *
from datasets.utils.tools import decode_segmap
import cv2


import os
os.environ["CUDA_VISIBLE_DEVICES"]='0'


class Tester(object):
    def __init__(self,  weight_path,
                        gpu_id,
                        visiual=False,
                        eval=False):
        self.device = gpu.select_device(gpu_id)
        self.weight_path = weight_path

        self.visiual = visiual
        self.eval = eval

        self.test_set = pascal.VOCSegmentation(base_size=cfg.TEST["BASE_SIZE"],
                                              crop_size=cfg.TEST["CROP_SIZE"],
                                              base_dir=cfg.DATA["TEST_DIR"],
                                              split='test')
        self.num_class = self.test_set.NUM_CLASSES
        self.test_loader = DataLoader(self.test_set,
                                     batch_size=cfg.TEST["BATCH_SIZE"],
                                     shuffle=False,
                                     num_workers=cfg.TEST["NUMBER_WORKERS"],
                                     pin_memory=False,
                                     drop_last=False)

        self.model = DeepLab(num_classes=self.num_class,
                             backbone="resnet",
                             output_stride=16,
                             sync_bn=False,
                             freeze_bn=False).to(self.device)

        self.evaluator = Evaluator(self.num_class)

        self.__load_model_weights(weight_path)

    def __load_model_weights(self, weight_path):
        print("loading weight file from : {}".format(weight_path))

        weight = os.path.join(weight_path)
        chkpt = torch.load(weight, map_location=self.device)
        self.model.load_state_dict(chkpt)
        print("loading weight file is done")
        del chkpt

    def test(self):
        self.model.eval()
        self.evaluator.reset()
        for i, sample in enumerate(self.test_loader):
            image, target = sample["image"], sample["label"]
            image = image.to(self.device)
            target = target.to(self.device)

            with torch.no_grad():
                out = self.model(image)

            pred = out.data.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            target = target.cpu().numpy()
            self.evaluator.add_batch(target, pred)

            if self.visiual:
                img_tmp = np.transpose(image.squeeze().data.cpu().numpy(), axes=[1,2,0])
                img_tmp *= (0.229, 0.224, 0.225)
                img_tmp += (0.485, 0.456, 0.406)
                img_tmp *= 255.0
                img_tmp = img_tmp[..., ::-1].astype(np.uint8)

                segmap_tmp = np.array(pred.squeeze()).astype(np.uint8)
                segmap = decode_segmap(segmap_tmp, dataset='pascal')
                segmap *= 255.0

                image_concat = np.concatenate((img_tmp, segmap[..., ::-1]))
                print("write image and sepmap : {}".format(i))

                cv2.imwrite("result/images/img_cat_{}.png".format(i), image_concat)


        if self.eval:
            Acc = self.evaluator.Pixel_Accuracy()
            Acc_class = self.evaluator.Pixel_Accuracy_Class()
            mIoU = self.evaluator.Mean_Intersection_over_Union()
            FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()

            # print
            print('*' * 20 + "Validate" + '*' * 20)
            print("Acc: {}\nAcc_class: {}\nmIoU: {}\nfwIoU: {}".
                  format(Acc, Acc_class, mIoU, FWIoU))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight_path', type=str, default='result/weights/best.pt', help='weight file path')
    parser.add_argument('--visiual', action='store_true', default=True,  help='visiual images')
    parser.add_argument('--eval', action='store_true', default=True, help="data augment flag")
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
    opt = parser.parse_args()

    Tester(weight_path=opt.weight_path,
            visiual=opt.visiual,
            eval=opt.eval,
            gpu_id=opt.gpu_id).test()
