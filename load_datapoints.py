import argparse
import numpy as np
import bz2
import pickle
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import yaml

def load_one_data_files(files):
    for file in files:
        data = {}
        with bz2.BZ2File(file, 'rb') as f:
            tmp_data = pickle.load(f)
        data['yaml'] = tmp_data[0]
        data['value_of_permuted_feild'] = tmp_data[1]
        data['coordinate'] = tmp_data[2]
        data['timeVSvmem']= tmp_data[3]
        
        yield file, data


data_path = 'Aditi Data/data_logs'
number_of_frames_to_analyse = 300 # or 0
save_frames_from_begining = False

files = [f for f in glob.glob(data_path + '/*.pkl')]
load_andler = load_one_data_files(files=files)

# load raw data
dataPoints = list()
dataPoint_file = list()
for file, dataPoint in tqdm(load_andler, total=len(files)):
    tmp_d = dataPoint['timeVSvmem'][-number_of_frames_to_analyse:, :]
    dataPoints.append(tmp_d)
    dataPoint_file.append(file)
    
    
# load normalize data

with bz2.BZ2File(
    'Aditi Data/Analysis/dataPoints_normalize_'
    + str(number_of_frames_to_analyse)
    + "_"
    + str(save_frames_from_begining)
    + ".pkl",
    "rb",
) as f:
    normalize_data = pickle.load(f)