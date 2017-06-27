import numpy as np
import sklearn.linear_model as linear

''' assumption:
1. ds format: 
    (timestamp, x, y, z, intensity)
2. one frame 
'''

TIME_SPEED_SCALE = 3600
def calibration_xpcd(ds, vx, vy, vz):
    start_time = np.min(ds[:, 0])
    delta_time = ds[:, 0] - start_time

    ds[:, 1] -= delta_time * vx / TIME_SPEED_SCALE
    ds[:, 2] -= delta_time * vy / TIME_SPEED_SCALE
    ds[:, 3] -= delta_time * vz / TIME_SPEED_SCALE
    return ds

def project_to(ds, target='xy'):
    if 'xy' == target:
        return ds[:, [1,2]]
    elif 'yz' == target:
        return ds[:, [2,3]]
    elif 'xz' == target:
        return ds[:,[1,3]]
    else:
        print('Not such target galaxy', target)

def floor(ds):
    value_threshold = 0.5 # 1m
    num_grid = 20
    xmax = np.min(ds[:, 0])
    xmin = np.max(ds[:, 0])
    grid_w = ( xmax - xmin) / num_grid
    ground_mean = np.zeros((num_grid))

    ''' grid id of each point '''
    grid_id = ((ds[:, 0] - xmin) / grid_w).astype(np.int16)

    ''' ground for grid '''
    for i in range(num_grid):
        pts = ds[grid_id == i]
        min = np.min(pts)
        limit = min + value_threshold
        # print("len of all pts: ", len(pts))
        pts = pts[pts[:, 1] < limit]
        # print("len of pts under limitation: ", len(pts))
        ground_mean[i] = np.mean(pts[:, 1])

    x = np.linspace(xmin, xmax, num_grid)
    x += grid_w/2
    # print("x-pos: ", x)
    # print("x-means: ", ground_mean)
    nds = np.stack((x, ground_mean), axis=1)

    coef = get_coef(nds)
    return coef

def get_coef(yz):
    y = yz[:, 0].reshape(-1,1)
    z = yz[:, 1].reshape(-1,1)

    model = linear.LinearRegression()
    model.fit(y, z)

    return model.coef_

def align_floor(ds, coef, axis = 'yz'):
    print("shape of ds: ", ds.shape)
    print("coef: ", coef)
    ds[:, 3] -= ds[:, 2] * coef

    return ds

def calibrate_floor(ds, distance_low = 15, distance_high = 150, plane = 'yz'):
    ds_in_plane = project_to(ds, plane)

    ''' use floor range, normally in (15, 150) '''
    ds_in_range = ds_in_plane[ds_in_plane[:, 0] < distance_high]
    ds_in_range = ds_in_range[ds_in_range[:, 0] > distance_low]

    ''' get floor '''
    coef = floor(ds_in_range)
    # print("floor coef: ", coef)

    # y = ds_in_plane[:, 0]
    # z = ds_in_plane[:, 1]
    #
    # plt.figure()
    # plt.scatter(y, z, s=0.5)
    # h = np.linspace(np.min(y), np.max(y), 20)
    # v = h * coef
    # plt.plot(h, v.squeeze())
    # plt.show()

    ds = align_floor(ds, coef.ravel())
    return ds

def distance_filter(d, low=5, high=150):
    # d = d[d[:, 3] > -1]
    d = d[d[:, 2] < high]
    d = d[d[:, 2] > low]
    return d

def speed_calibrate(d, speed_x=0, speed_y=0):
    # just for align
    # d_left = np.array([p for p in d if p[1] < 0])
    # d_right = np.array([p + [0,0,0,3,0] for p in d if p[1] > 0])
    # d = np.concatenate((d_left, d_right))
    # print("data: \n", d)

    ''' speed on X axis '''
    stime = np.min(d[:, 0])
    print("start time: ", stime)
    delta_time = d[:, 0] - stime

    ''' calibrate on X and Y axis '''
    spd_cal = lambda v : delta_time * (v / 3600000)

    if speed_x:
        delat_x = spd_cal(speed_x)
        delat_x += 6
        print("DELTA_X: ", delat_x)
        d[:,1] += delat_x
        # for i in range(len(d)):
        #     d[i][1] += delat_x[i]

    ''' speed on Y axis '''
    if speed_y:
        delta_y = spd_cal(speed_y)
        d[:,2] += delta_y

    ''' filtering '''
    d = distance_filter(d)

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
    return d