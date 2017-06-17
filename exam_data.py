import time
import numpy as np

from loader import *
from param import *
from filters import *

def plot_per_frame(pts):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = pts[:,1]
    y = pts[:,2]
    z = pts[:,3]
    scat = ax.scatter(x, y, z, c=y, s=1, cmap=cm.brg)
    # cb = plt.colorbar(scat)

    # plt.xlim(-3,5)
    # plt.ylim(3, 14)
    # ax.set_zlim(-2, 4)

    plt.show()
    pass

from mayavi.mlab import *
def show_per_frame(pts):
    # from vtk.api import tvtk
    # import mayavi
    # from mayavi.scripts import mayavi2

    x = pts[:,1]
    y = pts[:,2]
    z = pts[:,3]

    # colors = y/(y.max()-y.min()) + 0.5
    colors = z / (z.max() - z.min()) + 0.5

    obj = points3d(x, y, z, scale_factor=1)
    # obj.glyph.color_mode = 'color_by_vector'
    obj.glyph.scale_mode = 'scale_by_vector'
    obj.mlab_source.dataset.point_data.scalars = colors
    # print(result)
    axes(obj)
    show()


if __name__ == '__main__':
    ''' arguments '''
    arg = get_argument()
    if(arg.format == 'csv' and 'no_exist.csv' != arg.file ):
        d = load_cepton_csv(arg.file)
    else:
        d = load_file_with_id(arg.path, arg.id, format=arg.format)

    print("Shape of data: ", d.shape)

    # d = filters(d, speed_x=arg.vx, speed_y=arg.vy)
    # d = alpha(d)

    # plot_per_frame(d)
    # print(d)
    show_per_frame(d)
    pass
