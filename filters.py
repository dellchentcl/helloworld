import numpy as np

def distance_filter(d):
    # d = d[d[:, 3] > -1]
    d = d[d[:, 2] < 15]
    d = d[d[:, 2] > 2]
    return d

def filters(d, speed_x=0, speed_y=0):
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
        delta_y = spd_cal(speed_x)
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