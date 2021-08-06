
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
from tensorflow.keras.utils import to_categorical

from os import path

from captum.attr import IntegratedGradients

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
    parser.add_argument('--batch_size', type=int, default=50)
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
        ('batch_size', args.batch_size),
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

    logger.info('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(
        epoch, loss_meter.avg, accuracy))

    elapsed = time.time() - start
    logger.info('Elapsed {:.2f}'.format(elapsed))

    return accuracy


def data_parser(data_config):
    id_count = 2 # uid and rid
    ops_count = 1 # we experiment this for 1 operation
    metadata_count = 8 #in our experiment dataset (u4k-r4k-auth11k), each user/ resource has eight metadata
    
    trainDataFileName = data_config['train_data']
    testDataFileName = data_config['test_data']

    cols = id_count + (metadata_count * 2) + ops_count # <uid rid><8 user-meta and 8 resource-meta><1 ops>

    # load the training dataset
    train_raw_dataset = loadtxt(trainDataFileName, delimiter=' ', dtype=np.str)
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
    test_raw_dataset = loadtxt(testDataFileName, delimiter=' ', dtype=np.str)
    test_dataset = test_raw_dataset[:,2:cols] # TO SKIP UID RID

    np.random.shuffle(test_dataset)

    test_urp = test_dataset[:,0:metadata]
    test_operations = test_dataset[:,metadata:feature]
    test_operations = test_operations.astype('float16')

    if debug:
        print(train_urp[0])
        print(train_operations[0])

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

    x_train, x_test, y_train, y_test = data_parser(data_config)
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

    train_loader, test_loader = get_loader(optim_config['batch_size'],
                                           x_train, x_test, y_train, y_test)

    if debug:
        print('train_loader len', len(train_loader), 'test_loader', len(test_loader))
    
    model = load_model(config['model_config'])
    n_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    logger.info('n_params: {}'.format(n_params))

    criterion = nn.CrossEntropyLoss(size_average=True)

    # optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=optim_config['base_lr'])
    
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=optim_config['milestones'],
        gamma=optim_config['lr_decay'])
    
    model_path = os.path.join('neural_network', 'dlbac_alpha.pth')
    train_load_save_model(model, model_path)
    model.eval()
    
    # as interpretation experiment is not related to training, we
    # create both train_loader and test_loader to keep overall code reusable.
    # Here, both train_loader and test_loader contain same data (we take only one dataset as input). 
    # We experimented with samples from training dataset set.
    dataloader_iterator = iter(train_loader)

    integrated_gradients = IntegratedGradients(model)

    features = 16 #number of user and resource metadata
    aig = np.zeros(shape=(1,features))
    
    testdata, targets = next(dataloader_iterator)
    
    input_data = testdata
    outputs = model(input_data.float())
    
    if debug:
        print('Prediction result of first sample:', outputs[0])
    prediction_score, pred_label_idx = torch.max(outputs, dim=1)
    
    if debug:
        print('prediction_score:', prediction_score[0], ' pred_label:', pred_label_idx[0])
    attributions_ig = integrated_gradients.attribute(input_data.float(), target=pred_label_idx, n_steps=50)
    squz = attributions_ig.detach().numpy().sum(0)
    if debug:
        print('shape of squz', squz.shape)

    rows = squz.shape[0]
    cols = squz.shape[1]
    
    for row in range(rows):
        for col in range(cols):
            if squz[0][row][col] > 0:
                aig[0][row] = squz[0][row][col]
    
    result_file_path = os.path.join(outdir, 'global_interpret_result.txt')
    result_file = open(result_file_path, 'w+')
    
    result_file.write('Number of samples for global interpretation:%d\n\n' % (optim_config['batch_size']))
    
    if debug:
        print('\nRaw attribution information.\n')
        print(aig[0])
    
    result_file.write('\nRaw attribution information.\n')
    for meta in range(features):
        if meta < 8:
            result_file.write('umeta%d attribution: %s\n' % (meta, str(aig[0][meta])))
        else:
            result_file.write('rmeta%d attribution: %s\n' % (meta%8, str(aig[0][meta])))

    ig_attr_test_sum = aig
    ig_attr_test_norm_sum = ig_attr_test_sum / np.linalg.norm(ig_attr_test_sum, ord=1)
    
    if debug:
        print('Normalized attribution information.')
        print(ig_attr_test_norm_sum)
    
    result_file.write('\n\nNormalized attribution information.\n')
    for meta in range(features):
        if meta < 8:
            result_file.write('umeta%d attribution: %s\n' % (meta, str(ig_attr_test_norm_sum[0][meta])))
        else:
            result_file.write('rmeta%d attribution: %s\n' % (meta%8, str(ig_attr_test_norm_sum[0][meta])))
    
    result_file.close()
    print('Attribution information exported to: %s' % (result_file_path))


if __name__ == '__main__':
    main()


