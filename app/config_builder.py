import argparse

import _add_hrnet_lib # needed for import of hrnet library classes
from config import config
from config import update_config

def from_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    
    # hrnet arguments
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--testModel',
                        help='testModel',
                        type=str,
                        default='')
    # custom arguments
    parser.add_argument('--modelFamily',
                        help='modelFamily, default is hrnet. other otions are resnet18 and resnet152',
                        type=str,
                        default='hrnet')
    parser.add_argument('--transferEpochs',
                        help='transferEpochs, number of epcochs used for transfer learning',
                        type=int,
                        default=5)
    parser.add_argument('--transferBatchSize',
                        help='transferBatchSize, batch size used for transfer learning',
                        type=int,
                        default=5)
    parser.add_argument('--transferDataFolder',
                        help='transferDataFolder, default is hrnet. other otions are resnet18 and resnet152',
                        type=str,
                        default='../../data/heritage')
    parser.add_argument('--visualizeClasses',
                        help='visualizeClasses, shows images/classes from train and test set',
                        action='store_true')

    args = parser.parse_args()

    # hrnet config util
    update_config(config, args)

    # custom arguments
    config.defrost()
    if args.modelFamily:
        config.MODEL_FAMILY = args.modelFamily
    if args.transferEpochs:
        config.TRANSFER_EPOCHS = args.transferEpochs
    if args.transferBatchSize:
        config.TRANSFER_BATCH_SIZE = args.transferBatchSize
    if args.transferDataFolder:
        config.TRANSFER_DATA_FOLDER = args.transferDataFolder
    config.VISUALIZE_CLASSES = args.visualizeClasses

    config.freeze()

    return config
