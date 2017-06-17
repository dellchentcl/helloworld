import numpy as np

''' TODO: confirm the root cause '''
def alpha(d):
    d[:, 3] += d[:, 2] * 1.9 / 40
    return d

def distance_filter(d):
    # d = d[d[:, 3] > -3]
    # d = d[d[:, 2] < 150]
    # d = d[d[:, 2] < 15]
    # d = d[d[:, 2] > 2]
    d = d[d[:, 1] > -3]
    d = d[d[:, 1] < 10]
    return d

def half(d, side = 'left'):
    if 'left' == side:
        # nd = d[d[:, 1] < 0]
        nd = np.array([p for p in d if p[1] < 0])
    else:
        # nd = d[d[:, 1] > 0]
        nd = np.array([p for p in d if p[1] > 0])
    # print (nd)
    return nd

''' calibrate on X and Y axis '''
spd_cal = lambda v, dt: dt * (v / 3600000)

def cali_speed_x(d, delta_time, speed_x):
    if speed_x:
        delat_x = spd_cal(speed_x, delta_time)
        ''' TODO: think about it, it shall be out of vision '''
        # delat_x += 6
        print("DELTA_X: ", delat_x)
        d[:,1] -= delat_x
        # for i in range(len(d)):
        #     d[i][1] += delat_x[i]
    return d

def cali_speed_y(d, delta_time, speed_y):
    if speed_y:
        delta_y = spd_cal(speed_y, delta_time)
        d[:,2] -= delta_y
    return d

def cali_speed_z(d, delta_time, speed_z):
    if speed_z:
        delta_z = spd_cal(speed_z, delta_time)
        d[:,3] -= delta_z
    return d

def filters(d, speed_x=0, speed_y=0, speed_z=0):
    # just for align
    # d_left = np.array([p for p in d if p[1] < 0])
    # d_right = np.array([p + [0,0,0,3,0] for p in d if p[1] > 0])
    # d = np.concatenate((d_left, d_right))
    # print("data: \n", d)

    # d = half(d, 'right')

    d = alpha(d)

    ''' speed on X axis '''
    stime = np.min(d[:, 0])
    print("start time: ", stime)
    delta_time = d[:, 0] - stime

    d = cali_speed_x(d, delta_time, speed_x)
    d = cali_speed_y(d, delta_time, speed_y)
    d = cali_speed_z(d, delta_time, speed_z)

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