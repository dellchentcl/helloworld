import numpy as np
import struct
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import argparse
import pandas as pd
import struct
import os

from param import *
from loader import *

THRESHOLDE_LiDAR = -0.84

def plot_cpcb(x, y, z, c):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x,y,z, c=y, marker = '.', s=0.2)
    plt.show()


pause = False
get_df = lambda d: pd.DataFrame({"time": d[:, 0].ravel(), "x": d[:, 1].ravel(), "y": d[:, 2].ravel(),
                                 "z": d[:, 3].ravel(), "indensity": d[:, 4].ravel()})
def plot_dataset_with_animation(dataset):
    # filter = lambda data: [p for p in data if p[2]<20 ]
    data = dataset[0]
    df = get_df(data)

    def on_click(event):
        global pause

        if pause:
            ani.event_source.stop()
        else:
            ani.event_source.start()

        pause ^= True
        pass

    def on_key(event):
        global pause
        ani.event_source.start()
        pass

    def updater(num):
        # global graph
        global THRESHOLDE_LiDAR
        d = dataset[num]
        d = d[d[:,2] < 14]
        d = d[d[:,2] > 2]
        mn = np.mean(d[:,3])
        print("mean of z: ", mn)
        frame = get_df(d)
        graph._offsets3d = (frame.x, frame.y, frame.z)
        graph.changed()
        title.set_text('3D Test, id={}'.format(num))
        if(np.mean(d[:,3]) > THRESHOLDE_LiDAR):
            ani.event_source.stop()
            print("frame ID: ", num)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    title = ax.set_title('3D Test')
    graph = ax.scatter(df.x, df.y, df.z, c = df.y, s=0.5)

    ani = animation.FuncAnimation(fig, updater, 2600, interval=40, blit=False)
    # fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', lambda event: on_key(event))

    plt.xlim(-6,6)
    plt.ylim(3, 14)

    plt.show()
    pass

if __name__ == '__main__':
    arg = get_argument()
    ds = load_data_from_folder(arg.path, arg.format)
    print(ds.shape)
    # plot_cpcb(data[:, 1].ravel(), data[:,2].ravel(), data[:, 3].ravel(), data[:, 4].ravel())
    plot_dataset_with_animation(dataset=ds)
    # plot_per_frame(ds)
