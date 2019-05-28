import os
import pandas as pd

# Settings to tweak
# 0.78: 25, 0.3, 50
NUM_FILTERS = 15 # 25 best for LetNet # 60 best for original 
DROPOUT = 0.25   
BATCH = 20
LAMBDA = 0.1

EPOCHS = 100 #750
DIMS = 300
MIN_ACC = 0.70

# Files
TRAIN = os.path.join('data', 'a-r', 'train')
TEST = os.path.join('data', 'a-r', 'test')

CNN_WEIGHTS = os.path.join('data', 'weights', 'weights.h5')
CNN_BEST = os.path.join('data', 'weights', 'best.h5')
HIST = os.path.join('data', 'metadata', 'hist.txt')

# Partitioning data
PERCENT_TEST = 0.15
NUM_CLASSES = 2

NUM_TRAIN = 326
NUM_TEST = 39