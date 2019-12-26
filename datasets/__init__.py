from . import pascal
from torch.utils.data import DataLoader
import configs.deeplabv3plus_config_voc as cfg


def make_data_loader():

    if cfg.DATA["TYPE"] == 'pascal':
        train_set = pascal.VOCSegmentation(base_size=cfg.TRAIN["BASE_SIZE"],
                                           crop_size=cfg.TRAIN["CROP_SIZE"],
                                           base_dir=cfg.DATA["TRAIN_DIR"],
                                           split='train')
        val_set = pascal.VOCSegmentation(base_size=cfg.VAL["BASE_SIZE"],
                                         crop_size=cfg.VAL["CROP_SIZE"],
                                         base_dir=cfg.DATA["VAL_DIR"],
                                         split='val')
        # if args.use_sbd:
        #     sbd_train = sbd.SBDSegmentation(args, split=['train', 'val'])
        #     train_set = combine_dbs.CombineDBs([train_set, sbd_train], excluded=[val_set])

        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set,
                                  batch_size=cfg.TRAIN["BATCH_SIZE"],
                                  shuffle=True,
                                  num_workers=cfg.TRAIN["NUMBER_WORKERS"],
                                  pin_memory=True)
        val_loader = DataLoader(val_set,
                                batch_size=cfg.VAL["BATCH_SIZE"],
                                shuffle=False,
                                num_workers=cfg.VAL["NUMBER_WORKERS"],
                                pin_memory=True)
        test_loader = None

        return train_loader, val_loader, test_loader, num_class

    else:
        raise NotImplementedError