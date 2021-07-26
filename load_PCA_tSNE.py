import numpy as np
import bz2
import pickle
import matplotlib.pyplot as plt
import scipy.spatial as spatial
import mplcursors
from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting 
from matplotlib import cm


# 1. save image - save in subdirectory with title: filename_[n]
# vs save rotations option to save image rotated by 45 degrees all saved with filename_[n]

#show the graoh on the screen and save it to a file if the user asks to save on disk, with the parameters of the function
# if it is 3d, can the user view it from every direction. rotated 45 degrees
#resolution- automatically saved in a relatively high resolution
#no labeling for saving

ml_path = './Analysis/ML'
delta_path = './Analysis/Delta'

number_of_frames_to_analyse = 0
save_frames_from_begining = False

def plot_scatter(X, delta, title=None):
    fig = plt.figure()
    
    if X.shape[1] == 2: # 2D
        ax = plt.subplot(111)
        ax.scatter(X[:,0], X[:,1], alpha=0.5)
    elif X.shape[1] == 3: # 3D
        ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(X[:,0], X[:,1], X[:,2], alpha=0.5)
        ax.scatter(X[:,0], X[:,1], X[:,2],zdir='z',s=20,c=delta, depthshade=True)

    if title is not None:
        plt.title(title) 


# ---- tSNE
with bz2.BZ2File(ml_path + '/tSNE_' + str(number_of_frames_to_analyse) + '_normalize_' + str(save_frames_from_begining) + '.pkl', 'rb') as f:
    tSNE = pickle.load(f)

delta_csv = delta_path + '/delta_' + str(number_of_frames_to_analyse) + '_' + str(save_frames_from_begining) + '.csv'

delta = np.genfromtxt(delta_csv, delimiter=',')

plot_scatter(tSNE, delta)

plt.show()


# fig=plt.figure()

# ax=fig.gca(projection='3d')
# ax.view_init(elev=0, azim=0)


# x = tSNE[:,0]
# y = tSNE[:,1]
# z = tSNE[:,2]

##check that delta is normalized
# ax.scatter(x,y,z,zdir='z',s=20,c=delta, depthshade=True)
# mplcursors.cursor(hover=True)

# plt.show()


#----- PCA

# with bz2.BZ2File(ml_path + '/PCA_' + str(number_of_frames_to_analyse) + '_normalize_' + str(save_frames_from_begining) + '.pkl', 'rb') as f:
#     pca = pickle.load(f)

# plot_scatter(pca)
# plt.show()    

