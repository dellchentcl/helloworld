import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from loader import *
from param import *
from filters import *

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

    d = filters(d, speed_x=arg.vx, speed_y=arg.vy)

    plot_per_frame(d)
    pass
