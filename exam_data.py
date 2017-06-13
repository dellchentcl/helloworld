
import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from loader import *
from param import *

def plot_per_frame(pts):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = pts[:,1]
    y = pts[:,2]
    z = pts[:,3]
    scat = ax.scatter(x, y, z, c=y, s=2, cmap=cm.brg)
    # cb = plt.colorbar(scat)

    # plt.xlim(-3,5)
    plt.ylim(3, 14)
    # ax.set_zlim(-2, 4)

    plt.show()
    pass

if __name__ == '__main__':
    ''' arguments '''
    arg = get_argument()
    if(arg.format == 'csv' and 'no_exist.csv' != arg.file ):
        d = load_cepton_csv(arg.file)
    else:
        d = load_file_with_id(arg.path, arg.id, format=arg.format)

    # just for align
    # d_left = np.array([p for p in d if p[1] < 0])
    # d_right = np.array([p + [0,0,0,3,0] for p in d if p[1] > 0])
    # d = np.concatenate((d_left, d_right))

    # print("data: \n", d)
    #
    ''' timing calibration x value'''
    stime = np.min(d[:,0])
    print("start time: ", stime)
    delta_time = d[:,0] - stime
    # delta_time = delta_time.reshape(-1,1)
    delat_x = delta_time * ( arg.vx / 3600000) + 6
    print("DELTA_X: ", delat_x)

    # d[:,1] += delat_x
    for i in range(len(d)):
        d[i][1] += delat_x[i]



    # print("updated X: ", d)

    ''' filtering '''
    d = d[d[:, 2] < 14]
    d = d[d[:, 2] > 2]

    ''' merge duplicate data '''
    # gap = arg.gap
    # print("shape of d: ", d.shape)
    # print([p for p in d if p[1] < 0])
    # d_left = np.array( [p + [0, gap, 0, 0, 0] for p in d if p[1] < 0])
    # d_right = np.array([p for p in d if p[1] > 0])
    # d = np.concatenate((d_left, d_right))
    # print(np.array(d))

    ''' re-scaling '''
    # d = d * [1, 2, 1, 1, 1]

    plot_per_frame(d)
    pass
