# purpose: is it faster to try to look through all of the data and save all of the clicked points at once
# or is it better to try to locate the point you need on every click?

import numpy as np
import bz2
import pickle
import matplotlib.pyplot as plt
import _pickle as cpickle
import scipy.spatial as spatial
from matplotlib.widgets import Button
from datetime import datetime
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition
from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting 
from matplotlib import cm
import os
import pylab
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
import math
data_path = './Analysis'
ml_path = './Analysis/ML'
delta_path = './Analysis/Delta'

clicked_path = './Analysis/Clicked_Experiment'


def flush_clicked():
    # sort the clicked points
    num_points= 6805
    clicked_list = [1]



    line_num = 0
    current = 0
    clicked_fname = clicked_path + "/" + "clicked_" + datetime.now().strftime("%Y-%m-%d_%H.%M.%S") + ".pkl"
    done = False


    # raw_data_fname = data_path + "/dataPoints." + str(i) + ".pkl"
    # with bz2.BZ2File(raw_data_fname, 'rb') as f:
        # metadata = cpickle.load(f)
        
    
    with open(clicked_fname, "wb") as clicked_file:
        current = clicked_list[0]
        while not done:
            i = current * 4 // num_points
            raw_data_fname = data_path + "/dataPoints." + str(i) + ".pkl"
            with bz2.BZ2File(raw_data_fname, 'rb') as f:
                try: 
                    metadata = cpickle.load(f)
                    if (line_num == clicked_list[current]):
                        rdata = cpickle.load(f)
                        line_num += 1
                        cpickle.dump(rdata, clicked_file)
                        current += 1
                        if current >= len(clicked_list):
                            done = True
                    else:
                        cpickle.load(f)
                        line_num += 1
                except EOFError:
                    continue  


def read_clicked():
    for filename in os.listdir(os.getcwd()):
        with open(os.path.join(os.getcwd(), filename), 'rb') as f: # open in readonly mode
            print(filename)
            while True:
                try:
                    rdata = cpickle.load(f)
                except EOFError:
                    break

flush_clicked()
# read_clicked()