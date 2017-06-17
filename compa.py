from loader import *
from mayavi.mlab import *
import pcl
from pcl.registration import *

from param import *
from loader import *

def remove_ground(a, b):
    a = [p for p in a if p[1] > 50 or p[2] > 5 ]
    b = [p for p in b if p[1] > 50 or p[2] > 5 ]
    print(len(a))
    print(len(b))
    return a, b

''' -------------------test 1 ---------------- '''
def acc(folder, max_file):
    dataset = load_data_from_folder(folder, max_file=max_file)
    ds = np.zeros((0, 3))
    st = np.min(dataset[0][:,0])
    print("start time: ", st)

    for cloud in dataset:
        dt = cloud[:,0] - st
        # print("delta time: ", dt)
        cloud[:, 2] += dt * 30 / 3600000

        newcomer = cloud[:,[1,2,3]]
        ds = np.concatenate((ds, newcomer))
        # if 0 == len(ds):
        #     ds = np.concatenate((ds, newcomer))
        # else:
        #     ds = run_icp(newcomer, ds)
    # ds = ds[:,[1,2,3]]
    return ds

def test_model():
    arg = get_argument()
    # print(arg.files)

    ds = acc(arg.path, 30)
    print("Final Model Shape: ", ds.shape)
    plot_ds(ds)

''' -------------------test 2 ---------------- '''
def run_icp(a, b):
    # a, b = remove_ground(a, b)

    A = pcl.PointCloud()
    A.from_array(a.astype(np.float32))
    B = pcl.PointCloud()
    B.from_array(b.astype(np.float32))
    succ,T,est,fit= icp(A, B)
    # print(succ,T,est,fit)
    if not succ:
        print("ICP failed")
    else:
        print(T, fit)

    c = np.asarray(est)
    # print("shape of c", c.shape)

    ds = np.concatenate((b, c))

    return ds

def plot_ds(ds):
    # print(ds.shape)
    x = ds[:,0]
    y = ds[:,1]
    z = ds[:,2]
    colors = z / (z.max() - 30 - z.min()) + 0.5
    # colors = y / (y.max() - y.min()) + 0.5
    obj = points3d(x, y, z, scale_factor=0.5)
    obj.glyph.scale_mode = 'scale_by_vector'
    obj.mlab_source.dataset.point_data.scalars = colors

    axes(obj)
    show()

def test_icp():
    arg = get_argument()
    ''''''
    dataset = []
    for f in (arg.files).split(","):
        dataset.append(load_cpcd(arg.path+f))

    # ds = np.zeros((0, 5))
    # for d in dataset:
    #     ds = np.concatenate((ds, d))
    #
    # ds = ds[:,[1,2,3]]
    # plot_ds(ds)
    
    a = dataset[0]
    b = dataset[1]
    a = a[:, [1,2,3]]
    b = b[:, [1,2,3]]

    # print(np.min(a[:,2]), np.min(b[:,2]))

    ds = run_icp(a.astype(np.float32),b.astype(np.float32))
    plot_ds(ds)

''' -------------------test 3 ---------------- '''
def test_kd():
    arg = get_argument()
    dataset = []
    for f in (arg.files).split(","):
        dataset.append(load_cpcd(arg.path+f))
    a = dataset[0]
    b = dataset[1]
    a = a[:, [1,2,3]]
    b = b[:, [1,2,3]]

    # pc1 = pcl.PointCloud()
    # pc1.from_array(a.astype(np.float32))
    # pc2 = pcl.PointCloud()
    # pc2.from_array(b.astype(np.float32))
    # kd = pcl.KdTreeFLANN(pc1)
    # print(kd)
    # print(pc1)
    # print(pc2)

    pc1 = pcl.PointCloud(a.astype(np.float32))
    pc2 = pcl.PointCloud(b.astype(np.float32))
    kd = pc1.make_kdtree_flann()

    indices, sqr_distance = kd.nearest_k_search_for_cloud(pc2, 1)
    for i in range(pc1.size):
        print('squard distance is: ', sqr_distance[i, 0])
    ds = run_icp(a.astype(np.float32),b.astype(np.float32))
    plot_ds(ds)

def test_down_sampling():
    arg = get_argument()
    dataset = []
    for f in (arg.files).split(","):
        dataset.append(load_cpcd(arg.path+f))
    a = dataset[0]
    b = dataset[1]
    a = a[:, [1,2,3]]
    b = b[:, [1,2,3]]


if __name__ == '__main__':
    test_model()
    # test_icp()
    # test_kd()