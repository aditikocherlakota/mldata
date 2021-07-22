import numpy as np
import bz2
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting 

#color points using Analysis/Delta/dataPoints_0_False.pkl
#clickable point and label
#save point's original data to file
#click and save multiple points


ml_path = './Analysis/ML'
number_of_frames_to_analyse = 0
save_frames_from_begining = False

def plot_scatter(X, title=None):
    fig = plt.figure()
    
    if X.shape[1] == 2: # 2D
        ax = plt.subplot(111)
        ax.scatter(X[:,0], X[:,1], alpha=0.5)
    elif X.shape[1] == 3: # 3D
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X[:,0], X[:,1], X[:,2], alpha=0.5)
    
    if title is not None:
        plt.title(title) 


# ---- tSNE
with bz2.BZ2File(ml_path + '/tSNE_' + str(number_of_frames_to_analyse) + '_normalize_' + str(save_frames_from_begining) + '.pkl', 'rb') as f:
    tSNE = pickle.load(f)

plot_scatter(tSNE)
plt.show()



#----- PCA

with bz2.BZ2File(ml_path + '/PCA_' + str(number_of_frames_to_analyse) + '_normalize_' + str(save_frames_from_begining) + '.pkl', 'rb') as f:
    pca = pickle.load(f)

plot_scatter(pca)
plt.show()    
