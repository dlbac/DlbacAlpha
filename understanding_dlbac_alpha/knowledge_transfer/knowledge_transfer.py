
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

from numpy import loadtxt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model

from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import export_graphviz
from six import StringIO  
from IPython.display import Image  
import pydotplus

from os import path

logging.basicConfig(
    format='[%(asctime)s %(name)s %(levelname)s] - %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.DEBUG)
logger = logging.getLogger(__name__)

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

    parser.add_argument('--train_data', type=str, required=True)
    parser.add_argument('--test_data', type=str, required=True)
    parser.add_argument('--max_depth', type=int, default=8)
    parser.add_argument('--debug', type=str2bool, default=False)

    args = parser.parse_args()

    config = OrderedDict([
        ('train_data', args.train_data),
        ('test_data', args.test_data),
        ('max_depth', args.max_depth),
        ('outdir', 'result'),
        ('debug', args.debug),
    ])

    return config


def data_parser(config):
    id_count = 2 # uid and rid
    ops_count = 4
    metadata_count = 8 #in our experiment dataset (u4k-r4k-auth11k), each user/ resource has eight metadata
    
    trainDataFileName = config['train_data']
    testDataFileName = config['test_data']

    cols = id_count + (metadata_count * 2) + ops_count # <uid rid><8 user-meta and 8 resource-meta><4 ops>

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

    dt_x_train = train_urp #training data for the decision tree

    # load the testing dataset
    test_raw_dataset = loadtxt(testDataFileName, delimiter=' ', dtype=str)
    test_dataset = test_raw_dataset[:,2:cols] # TO SKIP UID RID

    np.random.shuffle(test_dataset)

    test_urp = test_dataset[:,0:metadata]
    test_operations = test_dataset[:,metadata:feature]
    test_operations = test_operations.astype('float16')

    dt_x_test = test_urp #test data for the decision tree

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

    return x_train, x_test, y_train, y_test, dt_x_train, dt_x_test


def main():
    # parse command line arguments
    config = parse_args()

    debug = config['debug']

    if debug:
        logger.info(json.dumps(config, indent=2))

    # create output directory
    outdir = config['outdir']
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    x_train, x_test, y_train, y_test, dt_x_train, dt_x_test = data_parser(config)
    if debug:
        print('x_train shape after return:', x_train.shape)
        print('y_train shape after return:', y_train.shape)
   
    
    model_path = os.path.join('neural_network', 'dlbac_alpha.hdf5')
    
    if os.path.exists(model_path):
        print('Loading trained model from {}.'.format(model_path))
        dlbac_alpha = load_model(model_path)
    else:
        print('No trained model found at {}.'.format(model_path))
        exit(0)

    dlbac_alpha_probs = dlbac_alpha.predict(x_train)

    dt_file_path = os.path.join(outdir, 'dlbac_alpha_decision_tree.png')
    
    dt_max_depth = config['max_depth']
    classifier = DecisionTreeRegressor(max_depth = dt_max_depth)
    classifier = classifier.fit(dt_x_train, dlbac_alpha_probs[:, 0])
    y_pred = classifier.predict(dt_x_test)    
    
    feature_cols = ["umeta0","umeta1","umeta2","umeta3","umeta4","umeta5","umeta6","umeta7","rmeta0","rmeta1","rmeta2","rmeta3","rmeta4","rmeta5","rmeta6","rmeta7"]

    dot_data = StringIO()
    export_graphviz(classifier, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True,
                feature_names = feature_cols)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    Image(graph.create_png())
    graph.write_png(dt_file_path)

    print('The decision tree is exported to: %s' % (dt_file_path))


if __name__ == '__main__':
    main()


