import numpy as np
import bz2
import pickle
import matplotlib.pyplot as plt
import scipy.spatial as spatial
from matplotlib.widgets import Button
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition
from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting 
from matplotlib import cm
import os
import time

# done 1. save image - save in subdirectory with title: filename_[n], use higher resolution
# done 2. save rotations - option to save image rotated by 45 degrees all saved with filename_[n]

# 3. for 2d graph, clickable/labeling functionality and also save image functionality,
# when doing clickable think about working with large amounts of data!

# 4. try using ckd trees to save points from 3d plot

# give up at some point and just save 3 graphs withh two variables

# 4. Try plotting large dataset

# left out- labeling for 3d graph, saving parameters of the function (??), colorbar, clickable points

ml_path = './Analysis/ML'
delta_path = './Analysis/Delta'
image_path = './Analysis/Images'

number_of_frames_to_analyse = 0
save_frames_from_begining = False

class Save:
    def __init__(self, ax):
        self.ax = ax;
        self.image_file = image_path + '/tSNE_' + str(number_of_frames_to_analyse) + '_normalize_' + str(save_frames_from_begining)
        self.file_type = '.png'
        if not os.path.isdir(image_path):
            os.makedirs(image_path)
        save_ax = plt.axes([0.7, 0.05, 0.1, 0.075])
        rotate_ax = plt.axes([0.81, 0.05, 0.1, 0.075])

        save_button = Button(save_ax, 'Save', color='grey')
        save_button.on_clicked(self.save)
        self.save_button = save_button

        rotation_button = Button(rotate_ax, 'Rotate', color='grey')
        rotation_button.on_clicked(self.rotate)
        self.rotation_button = rotation_button

    def save(self,event):
        print("saving")
        plt.savefig(self.image_file + self.file_type, dpi=1000,bbox_inches='tight')
        plt.draw()

    def rotate(self,event):
        print("rotating")
        for ii in range(0,360,45):
            self.ax.view_init(30, ii)
            plt.draw()
            plt.pause(1)
            plt.savefig(self.image_file + "_%d" % ii + self.file_type)
        plt.draw()

def plot_scatter(X, delta, title=None):
    if X.shape[1] == 2: # 2D
        ax = plt.subplot(111)
        ax.scatter(X[:,0], X[:,1], alpha=0.5)
    elif X.shape[1] == 3: # 3D
        ax = fig.add_subplot(111, projection='3d')
        data = Save(ax)
        ax.scatter(X[:,0], X[:,1], X[:,2],c=delta,s=2.0)
        return [data.rotation_button, data.save_button]

    if title is not None:
        plt.title(title) 


# ---- tSNE
fig = plt.figure()

with bz2.BZ2File(ml_path + '/tSNE_' + str(number_of_frames_to_analyse) + '_normalize_' + str(save_frames_from_begining) + '.pkl', 'rb') as f:
    tSNE = pickle.load(f)

delta_csv = delta_path + '/delta_' + str(number_of_frames_to_analyse) + '_' + str(save_frames_from_begining) + '.csv'

delta = np.genfromtxt(delta_csv, delimiter=',')

[b1, b2] = plot_scatter(tSNE, delta)


plt.show()

#----- PCA

# with bz2.BZ2File(ml_path + '/PCA_' + str(number_of_frames_to_analyse) + '_normalize_' + str(save_frames_from_begining) + '.pkl', 'rb') as f:
#     pca = pickle.load(f)

# plot_scatter(pca)
# plt.show()    

