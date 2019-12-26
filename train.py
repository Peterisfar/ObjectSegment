import utils.gpu as gpu
from models.loss.segmentation_loss import SegmentationLosses
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import argparse
from evals.evaluator import Evaluator
from utils.tools import *
import configs.deeplabv3plus_config_voc as cfg
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from models.nets.deeplab import *
from datasets import *


import os
os.environ["CUDA_VISIBLE_DEVICES"]='0'


class Trainer(object):
    def __init__(self,  weight_path, resume, gpu_id):
        init_seeds(1)
        init_dirs("result")

        self.device = gpu.select_device(gpu_id)
        self.start_epoch = 0
        self.best_mIoU = 0.
        self.epochs = cfg.TRAIN["EPOCHS"]
        self.weight_path = weight_path

        self.train_loader, self.val_loader, _, self.num_class = make_data_loader()

        self.model = DeepLab(num_classes=self.num_class,
                             backbone="resnet",
                             output_stride=16,
                             sync_bn= False,
                             freeze_bn=False).to(self.device)

        train_params = [{'params': self.model.get_1x_lr_params(), 'lr': cfg.TRAIN["LR_INIT"]},
                        {'params': self.model.get_10x_lr_params(), 'lr': cfg.TRAIN["LR_INIT"]*10}]

        self.optimizer = optim.SGD(train_params,
                                   momentum=cfg.TRAIN["MOMENTUM"],
                                   weight_decay=cfg.TRAIN["WEIGHT_DECAY"])

        self.criterion = SegmentationLosses().build_loss(mode=cfg.TRAIN["LOSS_TYPE"])

        self.scheduler = LR_Scheduler(mode=cfg.TRAIN["LR_SCHEDULER"],
                                      base_lr=cfg.TRAIN["LR_INIT"],
                                      num_epochs=self.epochs,
                                      iters_per_epoch=len(self.train_loader))
        self.evaluator = Evaluator(self.num_class)
        self.saver = Saver()
        self.summary = TensorboardSummary(os.path.join("result", "run"))

        if resume:
            self.__resume_model_weights()

    def __resume_model_weights(self):
        last_weight = os.path.join("result", "weights", "last.pt")
        chkpt = torch.load(last_weight, map_location=self.device)
        self.model.load_state_dict(chkpt['model'])

        self.start_epoch = chkpt['epoch'] + 1
        if chkpt['optimizer'] is not None:
            self.optimizer.load_state_dict(chkpt['optimizer'])
            self.best_mIoU = chkpt['best_mIoU']
        del chkpt

        print("resume model weights from : {}".format(last_weight))

    def __training(self, epoch):
        self.model.train()
        train_loss = 0.0
        for i, sample in enumerate(self.train_loader):
            image, target = sample["image"], sample["label"]
            image = image.to(self.device)
            target = target.to(self.device)

            self.scheduler(self.optimizer, i, epoch, self.best_mIoU)
            self.optimizer.zero_grad()

            out = self.model(image)
            loss = self.criterion(logit=out, target=target)
            loss.backward()
            self.optimizer.step()

            # Update running mean of tracked metrics
            train_loss = (train_loss*i + loss.item()) / (i+1)
            # Print or log
            if i%20==0:
                s = 'Epoch:[ {:d} | {:d} ]    Batch:[ {:d} | {:d} ]    loss: {:.4f}    lr: {:.6f}'.format(
                    epoch, self.epochs - 1, i, len(self.train_loader) - 1, train_loss,
                    self.optimizer.param_groups[0]['lr'])
                # self.logger.info(s)
                print(s)

            # Write
            global_step = i + len(self.train_loader) * epoch
            self.summary.writer.add_scalar('train/total_loss_iter', loss.item(), global_step)
            self.summary.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
            if i%(len(self.train_loader) // 10) == 0:
                self.summary.visualize_image(cfg.DATA["TYPE"], image, target, out, global_step)

        # Save last.pt
        if epoch <= 20:
            self.saver.save_checkpoint(
                state={'epoch': epoch,
                        'best_mAP': self.best_mIoU,
                        'model': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict()},
                is_best=False)

    def __validating(self, epoch):
        self.model.eval()
        self.evaluator.reset()
        test_loss = 0.0
        for i, sample in enumerate(self.val_loader):
            image, target = sample["image"], sample["label"]
            image = image.to(self.device)
            target = target.to(self.device)

            with torch.no_grad():
                out = self.model(image)
            loss = self.criterion(logit=out, target=target)
            test_loss += loss.item()

            pred = out.data.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            target = target.cpu().numpy()
            self.evaluator.add_batch(target, pred)

        #  calculate the index
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()

        # Write
        self.summary.writer.add_scalar('val/total_loss_epoch', test_loss, epoch)
        self.summary.writer.add_scalar('val/mIoU', mIoU, epoch)
        self.summary.writer.add_scalar('val/Acc', Acc, epoch)
        self.summary.writer.add_scalar('val/Acc_class', Acc_class, epoch)
        self.summary.writer.add_scalar('val/fwIoU', FWIoU, epoch)


        # Save
        is_best = False
        if mIoU > self.best_mIoU:
            self.best_mIoU = mIoU
            is_best = True
        self.saver.save_checkpoint(
            state={'epoch': epoch,
                   'best_mAP': self.best_mIoU,
                   'model': self.model.state_dict(),
                   'optimizer': self.optimizer.state_dict()},
            is_best=is_best)

        # print
        print('*' * 20 + "Validate" + '*' * 20)
        print("Acc: {}\nAcc_class: {}\nmIoU: {}\nfwIoU: {}\nLoss: {:.3f}\nbest_mIoU: {}".
              format(Acc, Acc_class, mIoU, FWIoU, test_loss, self.best_mIoU))


    def train(self):
        print(self.model)
        for epoch in range(self.start_epoch, self.epochs):
            self.__training(epoch)
            if epoch > -1:
                self.__validating(epoch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight_path', type=str, default='weight/darknet53_448.weights', help='weight file path')
    parser.add_argument('--resume', action='store_true',default=False,  help='resume training flag')
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
    opt = parser.parse_args()

    Trainer(weight_path=opt.weight_path,
            resume=opt.resume,
            gpu_id=opt.gpu_id).train()
