from param import *
from loader import *
from filters import *
import mayavi.mlab as mlab

if __name__ == '__main__':
    arg = get_argument()
    ds = load_data_from_folder(arg.path)

    print(ds.shape)
    data = ds[0]
    print(data.shape)

    l = mlab.points3d(data[:, 1], data[:, 2], data[:,3], scale_factor=0.5)
    colors = data[:,3]
    l.glyph.scale_mode = 'scale_by_vector'
    l.mlab_source.dataset.point_data.scalars = colors
    ms = l.mlab_source

    @mlab.animate(delay=100)
    def ani():
        for d in ds:
            colors = d[:, 3]
            # d = floor(d)
            ms.reset(x=d[:, 1], y=d[:, 2], z=d[:,3], scalars = colors)
            yield

    ani()
    mlab.show()

    pass