import mayavi.mlab as mlab
import matplotlib.pyplot as plt

def points2d(ds):
    plt.figure()
    plt.scatter(ds[:, 0], ds[:, 1], s=0.5)
    plt.show()

def points3d(ds):
    # mlab.test_plot3d()
    x = ds[:, 1]
    y = ds[:, 2]
    z = ds[:, 3]
    color = z
    pts = mlab.points3d(x, y, z, color, scale_factor = 0.2)
    pts.glyph.scale_mode = 'scale_by_vector'
    pts.glyph.color_mode = 'color_by_scalar'

    mlab.xlabel("x")
    mlab.ylabel("depth")
    mlab.zlabel("hight")
    mlab.show()
