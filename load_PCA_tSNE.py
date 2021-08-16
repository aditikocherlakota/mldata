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



# done 1. save image - save in subdirectory with title: filename_[n], use higher resolution
# done 2. save rotations - option to save image rotated by 45 degrees all saved with filename_[n]

# done 3. for 2d graph, clickable/labeling functionality and also save image functionality,
#5. done right filename for isngle graph saving


# 4. Try plotting large dataset
#6. 3d clickable point

# questions: 
#can i make saving faster by not using dump() 
# if not, investigate how to seek in a file
# check the time it takes to just pickle.load a file,m
# 1. what do the x y and z at bottom of screen come from?
# 2. what will happen if i use ax.transData.transform on the x and y data coordinates and then inv.transform on those same coordinates
# 3. what will happen if i look at all attributes of e/what they mean and figure out
# if i can use the inverse transform to go from display-->data
#difference between xdata ydata and x and y

# 4. try project 3d with 1,1,1 plotted. is 1,1,1 the data and x2 and y2 are display coords?? 
# if so is there an inverse to go from display coords to data
#if it is the display coords then is x2 y2 axes coords/something else? then it may not be useful

# if these fail hunt around more on internet


# observations: 
# 1. if i plot just one point: x y and z at bottom of the screen are the correct data values
# 2. e in on_motion gives me all i know about the clicked point. 


data_path = './Analysis'
ml_path = './Analysis/ML'
delta_path = './Analysis/Delta'

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


class ClickDotCursor_3D(FollowDotCursor):
    def __init__(self, ax, x, y, num_rdata_files, tolerance=5, formatter=fmt, offsets=(-20, 20)):
        FollowDotCursor.__init__(self, ax,x,y, tolerance, formatter, offsets)
        self.clicked_file = clicked_path + '/clicked_' + str(number_of_frames_to_analyse) + '_' + str(save_frames_from_begining)
        self.clicked_file_type = '.csv'
        self.num_rdata_files = num_rdata_files
        if not os.path.isdir(clicked_path):
            os.makedirs(clicked_path)
        clicked_ax = plt.axes([0.2, 0.05, 0.17, 0.075])

        clicked_button = Button(clicked_ax, 'Log Points', color='grey')
        clicked_button.on_clicked(self.flush_clicked)
        self.clicked_button = clicked_button

        cid = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.clicked_points = set()
    
    def on_click(self, event):
        # within = self.ax.get_position().contains(event.x,event.y)
        # if not within:
        #     return

        x, y = event.xdata, event.ydata
        _, idx = self.tree.query(self.scaled((x, y)), k=1, p=1)
        self.clicked_points.add(idx)

    def flush_clicked(self, event):
        # sort the clicked points
        clicked_list = sorted(self.clicked_points)
        print(clicked_list)
        line_num = 0
        current = 0
        clicked_fname = data_path + "/Clicked_Points/" + datetime.now().strftime("%Y-%m-%d_%H.%M.%S") + ".pkl"
        done = False
        for i in range(0, self.num_rdata_files):
            if done == True:
                print("Clickable points flushed")
                break
            raw_data_fname = data_path + "/dataPoints." + str(i) + ".pkl"
            with bz2.BZ2File(raw_data_fname, 'rb') as f:
                metadata = cpickle.load(f)
                with open(clicked_fname, "ab") as clicked_file:
                    while not done:
                        try:
                            if (line_num == clicked_list[current]):
                                rdata = cpickle.load(f)
                                # cpickle.dump(rdata, clicked_file)
                                current += 1
                                if current >= len(clicked_list):
                                    done = True
                            else:
                                cpickle.load(f)
                            line_num += 1
                        except EOFError:
                            break 
                # cpickle.dumps(yuh, clicked_file)

class ClickDotCursor(FollowDotCursor):
    def __init__(self, ax, x, y, num_rdata_files, tolerance=5, formatter=fmt, offsets=(-20, 20)):
        FollowDotCursor.__init__(self, ax,x,y, tolerance, formatter, offsets)
        self.clicked_file = clicked_path + '/clicked_' + str(number_of_frames_to_analyse) + '_' + str(save_frames_from_begining)
        self.clicked_file_type = '.csv'
        self.num_rdata_files = num_rdata_files
        if not os.path.isdir(clicked_path):
            os.makedirs(clicked_path)
        clicked_ax = plt.axes([0.2, 0.05, 0.17, 0.075])

        clicked_button = Button(clicked_ax, 'Log Points', color='grey')
        clicked_button.on_clicked(self.flush_clicked)
        self.clicked_button = clicked_button

        cid = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.clicked_points = set()
    
    def on_click(self, event):
        # within = self.ax.get_position().contains(event.x,event.y)
        # if not within:
        #     return

        x, y = event.xdata, event.ydata
        _, idx = self.tree.query(self.scaled((x, y)), k=1, p=1)
        self.clicked_points.add(idx)

    def flush_clicked(self, event):
        # sort the clicked points
        clicked_list = sorted(self.clicked_points)
        print(clicked_list)
        line_num = 0
        current = 0
        clicked_fname = data_path + "/Clicked_Points/" + datetime.now().strftime("%Y-%m-%d_%H.%M.%S") + ".pkl"
        done = False
        for i in range(0, self.num_rdata_files):
            if done == True:
                print("Clickable points flushed")
                break
            raw_data_fname = data_path + "/dataPoints." + str(i) + ".pkl"
            with bz2.BZ2File(raw_data_fname, 'rb') as f:
                metadata = cpickle.load(f)
                with open(clicked_fname, "ab") as clicked_file:
                    while not done:
                        try:
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
                            break        
class Save:
    def __init__(self, ax, Savefilename):
        self.ax = ax;
        self.Savefilename = Savefilename
        self.file_type = '.png'
        self.image_path = ml_path + "/PCA_" + str(number_of_frames_to_analyse) + " - " + self.Savefilename
        if not os.path.isdir(self.image_path):
            os.makedirs(self.image_path)
        save_ax = plt.axes([0.7, 0.05, 0.1, 0.075])

        save_button = Button(save_ax, 'Save', color='grey')
        save_button.on_clicked(self.save)
        self.save_button = save_button


    def save(self,event):
        print("saving")
        fname = self.image_path + "/" + self.Savefilename + "2D"
        plt.savefig(fname + ".jpeg", dpi=1000,bbox_inches='tight')
        plt.draw()



class Save_3D(Save):
    def __init__(self, ax, Savefilename):
        Save.__init__(self,ax, Savefilename)
        rotate_ax = plt.axes([0.81, 0.05, 0.1, 0.075])
        rotation_button = Button(rotate_ax, 'Rotate', color='grey')
        rotation_button.on_clicked(self.rotate)
        self.rotation_button = rotation_button
    def rotate(self,event):
        print("rotating")
        for ii in range(0,360,45):
            prefix = 30
            self.ax.view_init(prefix, ii)
            plt.draw()
            plt.pause(1)
            fname = self.image_path + "/" + self.Savefilename + "_" + '%s_%03d.jpeg' % (prefix, ii)
            plt.savefig(fname)
        plt.draw()
    def save(self,event):
        print("saving")
        fname = self.image_path + "/" + self.Savefilename + "_" + '%03d.jpeg' % (self.ax.azim)
        plt.savefig(fname, dpi=300)
        plt.draw()


    def on_motion(self, e):
        # move your mouse to (1,1,1), and e.xdata, e.ydata will be the same as x2, y2
        print(e)

def plot_scatter(X, delta, title=None, Savefilename=None):
    if X.shape[1] == 2: # 2D
        ax = plt.subplot(111)
        plt.subplots_adjust(bottom=0.2)
        ax.scatter(X[:,0], X[:,1], c=delta)
        # mpld3.show()
        # inv = ax.transData.inverted()
        # inv.transform((,  247.))
        data = Save(ax, Savefilename)
        cursor = ClickDotCursor(ax, X[:,0], X[:,1], 4, tolerance=20)
        return [data.save_button]
    elif X.shape[1] == 3: # 3D
        ax = fig.add_subplot(111, projection='3d')
        data = Save_3D(ax, Savefilename)
        # ax.scatter(X[1:5,0], X[1:5,1], X[1:5,2],c=delta[1:5],zdir='z')
        # ax.scatter(X[:,0], X[:,1], X[:,2],c=delta,s=2.0)
        # ax.scatter(X[1:3,0], X[1:3,1], X[1:3,2],c=delta[1:3])
        fig.canvas.mpl_connect('button_press_event', data.on_motion)
        ax.scatter(X[3,0], X[3,1], X[3,2],c=delta[1],zdir='z')
        # print(ax.transData.transform((X[1,0], X[1,1])))
        x2, y2, _ = proj3d.proj_transform(X[1,0], X[1,1], X[1,2], ax.get_proj())
        # print("lmao: ", x2, y2, uhh)
        # print("p1, p2, p3: ", X[1:5,0], X[1:5,1], X[1:5,2])
        return [data.rotation_button, data.save_button]

    if title is not None:
        plt.title(title) 

fig = plt.figure()

PCA_fname = ml_path + "/PCA_" + str(number_of_frames_to_analyse) +  '_' + str(save_frames_from_begining) + '_Volt.pkl'

with bz2.BZ2File(PCA_fname, 'rb') as f:
    PCA = pickle.load(f)

    X = list()
    with bz2.BZ2File(PCA_fname, 'rb') as f:
        while True:
            try:
                X.extend(pickle.load(f))
            except EOFError:
                break        
    X = np.array(X).reshape(-1, X[0].shape[0])

print(X.shape)

color_filename = ml_path + '/../Delta/delta_' + str(number_of_frames_to_analyse) + '_' + str(save_frames_from_begining) + '.csv'

delta = np.genfromtxt(color_filename, delimiter=',')

# [save,rotate] = plot_scatter(X, delta, title='All Cells Volt - unsupervise', Savefilename='All Cells Volt - unsupervise')
# [save] = plot_scatter(X[:,0:2], delta, title='All Cells Volt - unsupervise', Savefilename='All Cells Volt - unsupervise')

# [save, rotate] = plot_scatter(X, delta)

# plt.show()

pylab.show()
    
