import numpy as np
import bz2
import pickle
import matplotlib.pyplot as plt
import os



ml_path = 'Aditi Data/Analysis/ML'
number_of_frames_to_analyse = 0
save_frames_from_begining = False


def plot_scatter(PCA_Filename, title=None, Savefilename=None, path=None, colorFilename=None):
    global args
    
    X = list()
    with bz2.BZ2File(PCA_Filename, 'rb') as f:
        while True:
            try:
                X.extend(pickle.load(f))
            except EOFError:
                break        
    X = np.array(X).reshape(-1, X[0].shape[0])
    
    color_normalier = None
    if colorFilename is not None:
        import matplotlib
        color_data = np.loadtxt(colorFilename, delimiter=",")
        color_normalier = matplotlib.colors.Normalize(vmin=min(color_data), vmax=max(color_data))
        
    #fix size issues
    if X.shape[0] > color_data.shape[0]:
        X = X[:color_data.shape[0],:]
    else:
        color_data = color_data[:X.shape[0]]
    
    
    fig = plt.figure()
    fig.set_size_inches(30,18)
    
    if X.shape[1] == 2:  # 2D
        ax = plt.subplot(111)
        sc = ax.scatter(X[:, 0], X[:, 1], c=color_data, norm=color_normalier, alpha=0.5)
    elif X.shape[1] == 3:  # 3D
        ax = fig.add_subplot(111, projection="3d")
        sc = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color_data, norm=color_normalier, alpha=0.5)
        
    if color_data is not None:
        plt.colorbar(sc)

    if title is not None:
        plt.title(title)
    
    if Savefilename is None:
        plt.show()
    else:
        elevation = None
        angles = np.linspace(0, 360, 21)[:-1]  # A list of 20 angles between 0 and 360
        prefix = '_'
        path += " - " + title
        os.makedirs(path, exist_ok=True,)
        for i, angle in enumerate(angles):
            ax.view_init(elev=elevation, azim=angle)
            fname = path + "/" + Savefilename + '%s%03d.jpeg' % (prefix, i)
            plt.savefig(fname, dpi=300)
  

#----- PCA

with bz2.BZ2File(ml_path + '/PCA_' + str(number_of_frames_to_analyse) + '_' + str(save_frames_from_begining) + '_Volt.pkl', 'rb') as f:
    pca = pickle.load(f)

# -- All cells
path = ml_path + "/PCA_" + str(number_of_frames_to_analyse)
PCA_filename = path +  '_' + str(save_frames_from_begining) + '_Volt.pkl'
color_filename = ml_path + '/../Delta/delta_' + str(number_of_frames_to_analyse) + '_' + str(save_frames_from_begining) + '.csv'
plot_scatter(PCA_filename, path=path, Savefilename='All Cells Volt - unsupervise', colorFilename=color_filename, title='All Cells Volt - unsupervise')