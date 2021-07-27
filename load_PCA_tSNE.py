import numpy as np
import bz2
import pickle
import matplotlib.pyplot as plt
import scipy.spatial as spatial
from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting 

#color points using Analysis/Delta/dataPoints_0_False.pkl
#clickable point and label
#save point's original data to file
#click and save multiple points

number_of_frames_to_analyse = 0
save_frames_from_begining = False

pi = np.pi
cos = np.cos

def fmt(x, y):
    return 'x: {x:0.2f}\ny: {y:0.2f}'.format(x=x, y=y)

class FollowDotCursor(object):
    """Display the x,y location of the nearest data point.
    https://stackoverflow.com/a/4674445/190597 (Joe Kington)
    https://stackoverflow.com/a/13306887/190597 (unutbu)
    https://stackoverflow.com/a/15454427/190597 (unutbu)
    """
    def __init__(self, ax, x, y, tolerance=5, formatter=fmt, offsets=(-20, 20)):
        try:
            x = np.asarray(x, dtype='float')
        except (TypeError, ValueError):
            x = np.asarray(mdates.date2num(x), dtype='float')
        y = np.asarray(y, dtype='float')
        mask = ~(np.isnan(x) | np.isnan(y))
        x = x[mask]
        y = y[mask]
        self._points = np.column_stack((x, y))
        self.offsets = offsets
        y = y[np.abs(y-y.mean()) <= 3*y.std()]
        self.scale = x.ptp()
        self.scale = y.ptp() / self.scale if self.scale else 1
        self.tree = spatial.cKDTree(self.scaled(self._points))
        self.formatter = formatter
        self.tolerance = tolerance
        self.ax = ax
        self.fig = ax.figure
        self.ax.xaxis.set_label_position('top')
        self.dot = ax.scatter(
            [x.min()], [y.min()], s=130, color='green', alpha=0.7)
        self.annotation = self.setup_annotation()
        plt.connect('motion_notify_event', self)

    def scaled(self, points):
        points = np.asarray(points)
        return points * (self.scale, 1)

    def __call__(self, event):
        ax = self.ax
        # event.inaxes is always the current axis. If you use twinx, ax could be
        # a different axis.
        if event.inaxes == ax:
            x, y = event.xdata, event.ydata
        elif event.inaxes is None:
            return
        else:
            inv = ax.transData.inverted()
            x, y = inv.transform([(event.x, event.y)]).ravel()
        annotation = self.annotation
        x, y = self.snap(x, y)
        annotation.xy = x, y
        annotation.set_text(self.formatter(x, y))
        self.dot.set_offsets(np.column_stack((x, y)))
        bbox = self.annotation.get_window_extent()
        self.fig.canvas.blit(bbox)
        self.fig.canvas.draw_idle()

    def setup_annotation(self):
        """Draw and hide the annotation box."""
        annotation = self.ax.annotate(
            '', xy=(0, 0), ha = 'right',
            xytext = self.offsets, textcoords = 'offset points', va = 'bottom',
            bbox = dict(
                boxstyle='round,pad=0.5', fc='yellow', alpha=0.75),
            arrowprops = dict(
                arrowstyle='->', connectionstyle='arc3,rad=0'))
        return annotation

    def snap(self, x, y):
        """Return the value in self.tree closest to x, y."""
        dist, idx = self.tree.query(self.scaled((x, y)), k=1, p=1)
        try:
            return self._points[idx]
        except IndexError:
            # IndexError: index out of bounds
            return self._points[0]

def plot_scatter(X, delta, title=None, twoD=False):
    fig = plt.figure()
    if twoD:
        ax = plt.subplot(111)
        x= tSNE[:,0]
        y = tSNE[:,1]
        z = tSNE[:,2]
        ax.scatter(x,y,z,  c=delta)
    elif X.shape[1] == 2: # 2D
        ax = plt.subplot(111)
        x= tSNE[:,0]
        y = tSNE[:,1]
        ax.scatter(x,y,  c=delta)
        cursor = FollowDotCursor(ax, x, y, tolerance=20)
    elif X.shape[1] == 3: # 3D
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X[:,0], X[:,1], X[:,2])
    
    if title is not None:
        plt.title(title) 


ml_path = './Analysis/ML'
delta_path = './Analysis/Delta'

# ---- tSNE
with bz2.BZ2File(ml_path + '/tSNE_' + str(number_of_frames_to_analyse) + '_normalize_' + str(save_frames_from_begining) + '.pkl', 'rb') as f:
    tSNE = pickle.load(f)

delta_csv = delta_path + '/delta_' + str(number_of_frames_to_analyse) + '_' + str(save_frames_from_begining) + '.csv'

delta = np.genfromtxt(delta_csv, delimiter=',')

plot_scatter(tSNE[:, 0:2], delta)
plt.show()
    
#----- PCA

# with bz2.BZ2File(ml_path + '/PCA_' + str(number_of_frames_to_analyse) + '_normalize_' + str(save_frames_from_begining) + '.pkl', 'rb') as f:
#     pca = pickle.load(f)

# plot_scatter(pca)
# plt.show()    
