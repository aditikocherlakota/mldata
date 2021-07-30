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

#5. right filename for isngle graph saving

# left out- labeling for 3d graph, saving parameters of the function (??), colorbar, clickable points


data_path = './Analysis'
ml_path = './Analysis/ML'
delta_path = './Analysis/Delta'
image_path = './Analysis/Images'
clicked_path = './Analysis/Clicked_Points'

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
        x, y = event.xdata, event.ydata
        # within = ax.get_position().contains(event.x,event.y)
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

class ClickDotCursor(FollowDotCursor):
    def __init__(self, ax, x, y, tolerance=5, formatter=fmt, offsets=(-20, 20)):
        FollowDotCursor.__init__(self, ax,x,y, tolerance, formatter, offsets)
        self.clicked_file = clicked_path + '/clicked_' + str(number_of_frames_to_analyse) + '_' + str(save_frames_from_begining)
        self.clicked_file_type = '.csv'
        if not os.path.isdir(clicked_path):
            os.makedirs(clicked_path)
        clicked_ax = plt.axes([0.2, 0.05, 0.17, 0.075])

        clicked_button = Button(clicked_ax, 'Log Points', color='grey')
        clicked_button.on_clicked(self.flush_clicked)
        self.clicked_button = clicked_button

        cid = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.clicked_points = []
    
    def on_click(self, event):
        # within = self.ax.get_position().contains(event.x,event.y)
        # if not within:
        #     return

        x, y = event.xdata, event.ydata
        _, idx = self.tree.query(self.scaled((x, y)), k=1, p=1)
        self.clicked_points.append(idx)

    def flush_clicked(self, event):
        print("flush clicked")
        print(self.clicked_points)
class Save:
    def __init__(self, ax):
        self.ax = ax;
        self.image_file = image_path + '/tSNE_' + str(number_of_frames_to_analyse) + '_normalize_' + str(save_frames_from_begining)
        self.image_file_type = '.png'

        if not os.path.isdir(image_path):
            os.makedirs(image_path)


        save_ax = plt.axes([0.7, 0.05, 0.1, 0.075])

        save_button = Button(save_ax, 'Save', color='grey')
        save_button.on_clicked(self.save)
        self.save_button = save_button


    def save(self,event):
        print("saving")
        plt.savefig(self.image_file + "_" + str(self.ax.azim) + self.image_file_type, dpi=1000,bbox_inches='tight')
        plt.draw()



class Save_3D(Save):
    def __init__(self, ax):
        Save.__init__(self,ax)
        rotate_ax = plt.axes([0.81, 0.05, 0.1, 0.075])
        rotation_button = Button(rotate_ax, 'Rotate', color='grey')
        rotation_button.on_clicked(self.rotate)
        self.rotation_button = rotation_button
    def rotate(self,event):
        print("rotating")
        for ii in range(0,360,45):
            self.ax.view_init(30, ii)
            plt.draw()
            plt.pause(1)
            plt.savefig(self.image_file + "_%d" % ii + self.image_file_type)
        plt.draw()

def plot_scatter(X, delta, title=None, twoD=False):
    if X.shape[1] == 2: # 2D
        ax = plt.subplot(111)
        plt.subplots_adjust(bottom=0.2)
        ax.scatter(X[:,0], X[:,1], c=delta)
        data = Save(ax)
        cursor = ClickDotCursor(ax, X[:,0], X[:,1], tolerance=20)
        return [data.save_button]
    elif X.shape[1] == 3: # 3D
        ax = fig.add_subplot(111, projection='3d')
        data = Save_3D(ax)
        ax.scatter(X[:,0], X[:,1], X[:,2],c=delta,s=2.0)
        return [data.rotation_button, data.save_button]

    if title is not None:
        plt.title(title) 

# ---- tSNE
fig = plt.figure()

with bz2.BZ2File(data_path + '/dataPoints_' + str(number_of_frames_to_analyse) + '_' + str(save_frames_from_begining) + '.pkl', 'rb') as f:
    tSNE = pickle.load(f)


# np.savetxt("onethird_dataPoints_0_False.csv", tSNE[0][0:67], delimiter=",")


with bz2.BZ2File(ml_path + '/tSNE_' + str(number_of_frames_to_analyse) + '_normalize_' + str(save_frames_from_begining) + '.pkl', 'rb') as f:
    tSNE_norm = pickle.load(f)


print("end")
# delta_csv = delta_path + '/delta_' + str(number_of_frames_to_analyse) + '_' + str(save_frames_from_begining) + '.csv'

# delta = np.genfromtxt(delta_csv, delimiter=',')

# [save] = plot_scatter(tSNE_norm[:,0:2], delta)
# # [save, rotate] = plot_scatter(tSNE_norm, delta)

# plt.show()
    
#----- PCA

# with bz2.BZ2File(ml_path + '/PCA_' + str(number_of_frames_to_analyse) + '_normalize_' + str(save_frames_from_begining) + '.pkl', 'rb') as f:
#     pca = pickle.load(f)

# plot_scatter(pca)
# plt.show()    

