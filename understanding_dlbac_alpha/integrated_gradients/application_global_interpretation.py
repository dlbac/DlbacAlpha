
#!/usr/bin/env python
# coding: utf-8

import os
import time
import importlib
import json
from collections import OrderedDict
import logging
import argparse
import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torch.backends.cudnn
import torchvision.utils

from dataloader import get_loader

from numpy import loadtxt
#from tensorflow.python.keras.utils import to_categorical
from tensorflow.keras.utils import to_categorical

from os import path


logging.basicConfig(
    format='[%(asctime)s %(name)s %(levelname)s] - %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.DEBUG)
logger = logging.getLogger(__name__)

global_step = 0

debug = False

def str2bool(s):
    if s.lower() == 'true':
        return True
    elif s.lower() == 'false':
        return False
    else:
        raise RuntimeError('Boolean value expected')


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--depth', type=int, default=8)
    parser.add_argument('--debug', type=str2bool, default=False)

    args = parser.parse_args()

    model_config = OrderedDict([
        ('arch', 'resnet'),
        ('block_type', 'basic'),
        ('depth', args.depth),
        ('base_channels', 16),
        ('input_shape', (1, 3, 32, 32)),
        ('n_classes', 2),
    ])

    optim_config = OrderedDict([
        ('epochs', 0),
        ('batch_size', 16),
        ('base_lr', 1e-3),
        ('weight_decay', 1e-4),
        ('milestones', json.loads('[20, 30, 40]')),
        ('lr_decay', 0.1),
    ])

    data_config = OrderedDict([
        ('train_data', args.data),
        ('test_data', args.data),
    ])

    run_config = OrderedDict([
        ('seed', 17),
        ('outdir', 'result'),
        ('debug', args.debug),
    ])

    config = OrderedDict([
        ('model_config', model_config),
        ('optim_config', optim_config),
        ('data_config', data_config),
        ('run_config', run_config),
    ])

    return config


def load_model(config):
    module = importlib.import_module(config['arch'])
    Network = getattr(module, 'Network')
    return Network(config)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, num):
        self.val = val
        self.sum += val * num
        self.count += num
        self.avg = self.sum / self.count


def test(epoch, model, criterion, test_loader, run_config):
    
    if debug:
        logger.info('Test {}'.format(epoch))
    model = model.float()
    model.eval()

    loss_meter = AverageMeter()
    correct_meter = AverageMeter()
    start = time.time()
    for step, (data, targets) in enumerate(test_loader):

        with torch.no_grad():
            outputs = model(data.float())
        
        loss = criterion(outputs, targets)

        _, preds = torch.max(outputs, dim=1)

        loss_ = loss.item()
        correct_ = preds.eq(targets).sum().item()
        num = data.size(0)

        loss_meter.update(loss_, num)
        correct_meter.update(correct_, 1)

    accuracy = correct_meter.sum / len(test_loader.dataset)

    if debug:
        logger.info('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(
        epoch, loss_meter.avg, accuracy))

    elapsed = time.time() - start
    if debug:
        logger.info('Elapsed {:.2f}'.format(elapsed))

    return accuracy


def data_parser(data_config, no_of_changed_meta):
    id_count = 2 # uid and rid
    ops_count = 1 # we experiment this for 1 operation
    metadata_count = 8 #in our experiment dataset (u4k-r4k-auth11k), each user/ resource has eight metadata
    
    trainDataFileName = data_config['train_data']
    testDataFileName = data_config['test_data']

    cols = id_count + (metadata_count * 2) + ops_count # <uid rid><8 user-meta and 8 resource-meta><1 ops>

    # load the training dataset
    train_raw_dataset = loadtxt(trainDataFileName, delimiter=' ', dtype=str)
    train_dataset = train_raw_dataset[:,2:cols] # TO SKIP UID RID

    np.random.shuffle(train_dataset)

    feature = train_dataset.shape[1]
    if debug:
        print('Features:', feature)
    metadata = feature - ops_count

    train_urp = train_dataset[:,0:metadata]
    train_operations = train_dataset[:,metadata:feature]
    train_operations = train_operations.astype('float16')

    # load the testing dataset
    test_raw_dataset = loadtxt(testDataFileName, delimiter=' ', dtype=str)
    test_dataset = test_raw_dataset[:,2:cols] # TO SKIP UID RID

    #np.random.shuffle(test_dataset)

    test_urp = test_dataset[:,0:metadata]
    
    if debug:
        print('metadata value of first sample before replacing metadata value')
        print(test_urp[0])
    
    #modify test_data to replace metadata values with significant one
    if debug:
        print('Number of Changed Metadata: %d' % (no_of_changed_meta))

# for the changed value, we randomly select a tuple (uid:4246, rid:4435) with grant access on op1
    if no_of_changed_meta == 1:
        test_urp[:, 10] =  5
    elif no_of_changed_meta == 2:
        test_urp[:, 10] =  5
        test_urp[:, 2] =  5
    elif no_of_changed_meta == 3:
        test_urp[:, 10] =  5
        test_urp[:, 2] =  5
        test_urp[:, 4] =  44
    elif no_of_changed_meta == 4:
        test_urp[:, 10] =  5
        test_urp[:, 2] =  5
        test_urp[:, 4] =  44
        test_urp[:, 1] =  84
    elif no_of_changed_meta == 5:
        test_urp[:, 10] =  5
        test_urp[:, 2] =  5
        test_urp[:, 4] =  44
        test_urp[:, 1] =  84
        test_urp[:, 8] =  30
    elif no_of_changed_meta == 6:
        test_urp[:, 10] =  5
        test_urp[:, 2] =  5
        test_urp[:, 4] =  44
        test_urp[:, 1] =  84
        test_urp[:, 8] =  30
        test_urp[:, 13] =  105 #tuples started loosing access again at this stage
    
    if debug:
        print('metadata value of first sample after replacing metadata value')
        print(test_urp[0])

    test_operations = test_dataset[:,metadata:feature]
    test_operations = test_operations.astype('float16')

    ############### ENCODING ##############
    train_urp = to_categorical(train_urp)
    if debug:
        print('shape of Train-URP after encoding')
        print(train_urp.shape)

    test_urp = to_categorical(test_urp)
    if debug:
        print('shape of Test-URP after encoding')
        print(test_urp.shape)
    ############### End of Encoding #######

    x_train = train_urp
    x_test = test_urp
    if debug:
        print('x_train shape:', x_train.shape)
        print('x_test shape:', x_test.shape)

    y_train = train_operations
    y_test = test_operations

    if debug:
        print(x_train.shape[0], 'train samples')
        print(x_test.shape[0], 'test samples')
        print('y_train shape:', y_train.shape)

    return x_train, x_test, y_train, y_test


def train_load_save_model(model_obj, model_path):
    if path.isfile(model_path):
        if debug:
            print('Loading pre-trained model from: {}'.format(model_path))
        checkpoint = torch.load(model_path)
        model_obj.load_state_dict(checkpoint['state_dict'])
        if debug:
            print('model loading done!')


def main():
    # parse command line arguments
    config = parse_args()

    run_config = config['run_config']
    optim_config = config['optim_config']
    data_config = config['data_config']

    debug = run_config['debug']

    if debug:
        logger.info(json.dumps(config, indent=2))

    # set random seed
    seed = run_config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # create output directory
    outdir = run_config['outdir']
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # save config as json file in output directory
    outpath = os.path.join(outdir, 'config.json')
    with open(outpath, 'w') as fout:
        json.dump(config, fout, indent=2)

    max_metadata_change = 7

    result_file_path = os.path.join(outdir, 'application_global_interpret_result.txt')
    result_file = open(result_file_path, 'w+')

    for no_of_changed_meta in range(max_metadata_change):
        if debug:
            print('Number of Changed Metadata: %d' % (no_of_changed_meta))
        result_file.write('Number of Changed Metadata: %d \n' % (no_of_changed_meta))
        
        x_train, x_test, y_train, y_test = data_parser(data_config, no_of_changed_meta)
        if debug:
            print('x_train shape after return:', x_train.shape)
            print('y_train shape after return:', y_train.shape)
   
        model_config = config['model_config']
        if debug:
            print('before assigning, default input shape', model_config['input_shape'])
    
        input_shape = x_train[0].reshape((1,1,)+x_train[0].shape)
        model_config['input_shape'] = input_shape.shape
        if debug:
            print('model config input shape', model_config['input_shape'])

        # there is no use of train_loader 
        train_loader, test_loader = get_loader(optim_config['batch_size'],
                                           x_train, x_test, y_train, y_test)

        if debug:
            print('train_loader len', len(train_loader), 'test_loader', len(test_loader))
    
        model = load_model(config['model_config'])
        n_params = sum([param.view(-1).size()[0] for param in model.parameters()])
        if debug:
            logger.info('n_params: {}'.format(n_params))

        criterion = nn.CrossEntropyLoss(size_average=True)

    	# evaluate the performance
        model_path = os.path.join(outdir, 'model_state.pth')
        train_load_save_model(model, model_path)
        model.eval()
        accuracy = test(1, model, criterion, test_loader, run_config)
        if debug:
            print('Percentage of tuples with deny access: %f' % (accuracy * 100))
            print('Percentage of tuples receiving grant access: %f' % ((1-accuracy) * 100))
        result_file.write('Current percentage of tuples with deny access: %f \n' % (accuracy * 100))
        result_file.write('Percentage of tuples receiving grant access: %f \n\n' % ((1-accuracy) * 100))

    print('The outputs of global interpretation experiment are exported to: %s' % (result_file_path))
    result_file.close()


if __name__ == '__main__':
    main()


