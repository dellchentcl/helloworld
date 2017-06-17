import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import pandas as pd
from matplotlib import cm

from loader import *
from param import *
from extract_wqfmt import *
from filters import *

def build_model(data_set):
    ds = data_set[0]
    for i in range(len(data_set)):
        if i%3 == 0:
            ds = np.concatenate((ds, data_set[i]))
    return ds

def plot_model(pts, ys, ye):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    pts = pts[pts[:,2] > ys]
    pts = pts[pts[:,2] < ye]

    x = pts[:, 1]
    y = pts[:, 2]
    z = pts[:, 3]
    scat = ax.scatter(x, y, z, c=z, s=0.5, cmap=cm.brg)
    # cb = plt.colorbar(scat)

    # plt.xlim(-5,5)
    # plt.ylim(3, 14)
    # ax.set_zlim(-2, 4)

    plt.show()
    pass

FMT_CPCB = '<qffff'
def generate_pcl(fn, pts, ys, ye):

    dw = pts [ pts[:, 2] > ys ]
    dw = dw [pts[:, 2] < ye]

    print("shape of DW: ", dw.shape)

    with open(fn, 'wb') as of:
        for i in range(len(dw)):
            rec = dw[i]
            # print(*dw)
            # of.write(struct.pack(FMT_CPCB, rec[0], rec[1], rec[2], rec[3], rec[4]))
            # of.write(struct.pack(FMT_CPCB, *rec))
            of.write(rec)
    pass

if __name__ == '__main__':
    arg = get_argument()
    data_set = load_data_from_folder(arg.path)
    # print("Shape of data_set;", data_set.shape)

    ds = build_model(data_set)
    print("shape of all data set: ", ds.shape)
    # print("DATA: ", ds)

    ds = filters(ds, speed_x=arg.vx, speed_y=arg.vy, speed_z=arg.vz)

    plot_model(ds, arg.sy, arg.ey)
    # generate_pcl(arg.opcl, ds, arg.sy, arg.ey)

    # plot_acc_frames(data_set, arg.num_of_frames, arg.vx, arg.vy, arg.vz)

    pass