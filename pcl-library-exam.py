#! /home/dellchen/anaconda3/envs/py27/bin/python

from loader import *
from mayavi.mlab import *
import pcl
from pcl.registration import *
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

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

    ds = acc(arg.path, 0)
    print("Final Model Shape: ", ds.shape)
    plot_ds(ds)

    ds2d = ds[:, [0,1]]
    return ds2d

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

def plot_ds(ds, name = "None"):
    # print(ds.shape)
    x = ds[:,0]
    y = ds[:,1]
    z = ds[:,2]
    colors = z / (z.max() - 30 - z.min()) + 0.5
    # colors = y / (y.max() - y.min()) + 0.5
    obj = points3d(x, y, z, scale_factor=0.2)
    obj.glyph.scale_mode = 'scale_by_vector'
    obj.mlab_source.dataset.point_data.scalars = colors

    axes(obj)
    # text3d(0,0,20, title)
    title(name)
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

def detect_3rd():
    ds2d = test_model()
    plt.figure()
    plt.scatter(ds2d[:,0], ds2d[:, 1])
    plt.show()
    pass

''' -------------------test 4 ---------------- '''
def get_ioliers(cloud):
    fil = cloud.make_statistical_outlier_filter()
    fil.set_mean_k(50)
    fil.set_std_dev_mul_thresh(0.5)
    il = fil.filter()
    fil.set_negative(True)
    ol = fil.filter()
    return il, ol

def detect_plane():
    arg = get_argument()
    ds = load_cpcd(arg.file)
    ds = ds[:,[1,2,3]]
    print(ds.shape)
    ''' 1. original data '''
    plot_ds(ds, "original data")

    ''' PCA '''
    pca = PCA (n_components= 3)
    pca.fit(ds)
    new_pca = pca.transform(ds)
    plot_ds(new_pca, 'PCA')

    cloud = pcl.PointCloud(ds.astype(np.float32))

    il, ol = get_ioliers(cloud)
    il_array = il.to_array()
    plot_ds(il_array, "inliers")

    ol_array = ol.to_array()
    plot_ds(ol_array,"outliers")

    # cloud_array = cloud.to_array()
    # plot_ds(cloud_array)

    # fil = cloud.make_passthrough_filter()
    # fil.set_filter_field_name('z')
    # fil.set_filter_limits(0, 1.5)
    # cloud_filtered = fil.filter()
    # print(cloud_filtered.size)
    # filtered_array = cloud_filtered.to_array()
    # plot_ds(filtered_array)

    seg = cloud.make_segmenter_normals(ksearch=50)
    seg.set_optimize_coefficients(True)
    seg.set_model_type(pcl.SACMODEL_NORMAL_PLANE)
    seg.set_normal_distance_weight(2)
    seg.set_method_type(pcl.SAC_RANSAC)
    seg.set_max_iterations(100)
    seg.set_distance_threshold(0.2)
    indices, model = seg.segment()
    print(indices, model)

    ''' 2. extract plane '''
    cloud_plane = cloud.extract(indices, negative=False)
    cloud_plane.to_file("table_scene_mug_stereo_textured_plane.pcd")
    # print(cloud_plane)
    plane_array = cloud_plane.to_array()
    print(plane_array.shape)
    plot_ds(plane_array, "plane")

    ''' 3. leaving loud '''
    cloud_cyl = cloud.extract(indices, negative=True)
    cc_array = cloud_cyl.to_array()
    plot_ds(cc_array, "leaving data for TURE")

    ''' 4. extract plane again '''
    seg = cloud_cyl.make_segmenter_normals(ksearch=50)
    seg.set_optimize_coefficients(True)
    seg.set_model_type(pcl.SACMODEL_NORMAL_PLANE)
    seg.set_normal_distance_weight(3)
    seg.set_method_type(pcl.SAC_RANSAC)
    seg.set_max_iterations(100)
    seg.set_distance_threshold(1)
    indices, model = seg.segment()
    print(indices, model)
    cloud_cyl = cloud.extract(indices, negative=False)
    cc_array = cloud_cyl.to_array()
    plot_ds(cc_array, "extract plane again")

    ''' 5. extract cylinder '''
    seg = cloud_cyl.make_segmenter_normals(ksearch=50)
    seg.set_optimize_coefficients(True)
    seg.set_model_type(pcl.SACMODEL_CYLINDER)
    seg.set_normal_distance_weight(1)
    seg.set_method_type(pcl.SAC_RANSAC)
    seg.set_max_iterations(10000)
    seg.set_distance_threshold(0.5)
    seg.set_radius_limits(0, 0.5)
    indices, model = seg.segment()

    # print(model)
    cloud_cylinder = cloud_cyl.extract(indices, negative=False)
    cloud_cylinder.to_file("table_scene_mug_stereo_textured_cylinder.pcd")
    cyl_array = cloud_cylinder.to_array()
    print(cyl_array.shape)
    plot_ds(cyl_array, "cylinder")

    pass

def test_plane2(c1, c2):
    arg = get_argument()
    print(arg.files)
    if arg.files is not "":
        ds = load_files(arg.files)
    else:
        ds = load_cpcd(arg.file)
    ds = ds[:,[1,2,3]]

    x = ds[:, 0]
    y = ds[:, 1]
    z = ds[:, 2]

    xmin = np.min(x)
    xmax = np.max(x)
    ymin = np.min(y)
    ymax = np.max(y)
    # zmin = np.min(z)
    # zmax = np.max(z)

    x = x.reshape(-1,1)
    y = y.reshape(-1,1)
    z = z.reshape(-1,1)

    # z -= (x - xmin) * c1
    z -= (y - ymin) * c2

    ds = np.concatenate((x,y,z), axis=1)
    print("shape of new ds: ", ds.shape)

    ''' 1. original data '''
    plot_ds(ds, "original data")

    ''' segment for plane '''
    cloud = pcl.PointCloud(ds.astype(np.float32))
    seg = cloud.make_segmenter_normals(ksearch=50)
    seg.set_optimize_coefficients(True)
    seg.set_model_type(pcl.SACMODEL_NORMAL_PLANE)
    seg.set_normal_distance_weight(1.1)
    seg.set_method_type(pcl.SAC_RANSAC)
    seg.set_max_iterations(100)
    seg.set_distance_threshold(0.3)
    indices, model = seg.segment()

    ''' 2. extract plane '''
    cloud_plane = cloud.extract(indices, negative=False)
    cloud_plane.to_file("table_scene_mug_stereo_textured_plane.pcd")
    # print(cloud_plane)
    plane_array = cloud_plane.to_array()
    print(plane_array.shape)
    plot_ds(plane_array, "Plane")

    il, ol = get_ioliers(cloud_plane)
    il_array = il.to_array()
    plot_ds(il_array, "inliers")

    il2, ol2 = get_ioliers(il)
    il_array = il2.to_array()
    plot_ds(il_array, "inliers; inliers")

    il3, ol3 = get_ioliers(il)
    il_array = il3.to_array()
    plot_ds(il_array, "inliers; inliers' inliers")

    ol_array = ol.to_array()
    plot_ds(ol_array,"outliers")

    '''3. extract None-plane '''
    none_plane = cloud.extract(indices, negative=True)
    cc_array = none_plane.to_array()
    plot_ds(cc_array, "None-Plane")
    pass

def plot_scatter(a, b, title="none"):
    plt.figure()
    plt.scatter(a, b, c= b)
    plt.title(title)
    plt.show()

from sklearn import linear_model
def regress(x,z):
    xmin = np.min(x)
    xmax = np.max(x)
    linreg = linear_model.LinearRegression()
    print("shape of x/z: ", x.shape, z.shape)
    linreg.fit(x.reshape(-1,1), z.reshape(-1,1))

    test_x = np.linspace(xmin, xmax, 20)
    test_z = linreg.predict(test_x.reshape(-1,1))
    print('Coefficients: %f\n'% linreg.coef_)

    # plt.figure()
    # plt.scatter(x,z,c=z)
    # plt.plot(test_x, test_z)
    # plt.show()
    return linreg.coef_

def project2_xz_yz():
    # box_size = 0.2

    arg = get_argument()
    print(arg.files)
    if arg.files is not "":
        ds = load_files(arg.files)
    else:
        ds = load_cpcd(arg.file)

    ds = ds[:,[1,2,3]]

    x = np.array(ds[:, 0])
    y = np.array(ds[:, 1])
    z = np.array(ds[:, 2])

    ''' linear regression on x-z, y/z '''
    coef_xz = regress(x,z)
    coef_yz = regress(y,z)

    ''' plot on x/z '''
    # plot_scatter(x, z, "x-z")
    # plot_scatter(x, y, "x-y")
    # plot_scatter(y, z, "y-z")

    # x_nums = ((xmax - xmin) / box_size).astype(np.int16) + 1
    # z_nums = ((zmax - zmin) / box_size).astype(np.int16) + 1

    return coef_xz, coef_yz

if __name__ == '__main__':
    # detect_3rd()
    # test_icp()
    # test_kd()
    # detect_plane()

    # c1, c2 = project2_xz_yz()
    test_plane2(0, 0)

