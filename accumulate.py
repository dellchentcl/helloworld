import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import pandas as pd

from loader import *
from param import *
from extract_wqfmt import *
from filters import *

pause = False
get_df = lambda d: pd.DataFrame({"time": d[:, 0].ravel(), "x": d[:, 1].ravel(), "y": d[:, 2].ravel(),
                                 "z": d[:, 3].ravel(), "indensity": d[:, 4].ravel()})

# MAX_FRAME_TO_ACCUMULATED = 5
def get_frames(dataset, num, num_of_frames):
    if 0 == num:
        return dataset[0]

    sf = 0
    if num > num_of_frames:
        sf = num - num_of_frames

    ds = dataset[sf]
    for i in range(sf, num):
        ds = np.concatenate((ds, dataset[i]))

    return ds

def plot_acc_frames(dataset, num_of_frames, speed_x = 0, speed_y = 0, speed_z = 0):
    # filter = lambda data: [p for p in data if p[2]<20 ]
    data = dataset[0]

    df = get_df(data)

    def on_click(event):
        global pause

        if pause:
            ani.event_source.stop()
        else:
            ani.event_source.start()

        pause ^= True
        pass

    def on_key(event):
        ani.event_source.start()
        pass

    def updater(num):
        # global graph
        global THRESHOLDE_LiDAR
        d = get_frames(dataset, num, num_of_frames)

        # d = distance_filter(d)
        d = filters(d, speed_x, speed_y, speed_z)

        # mn = np.mean(d[:,3])
        # print("mean of z: ", mn)
        frame = get_df(d)
        graph._offsets3d = (frame.x, frame.y, frame.z)
        graph.changed()
        title.set_text('3D Test, id={}'.format(num))
        ani.event_source.stop()
        print("frame ID: ", num)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    title = ax.set_title('3D Test')
    graph = ax.scatter(df.x, df.y, df.z, c = df.z, s=0.2)

    ani = animation.FuncAnimation(fig, updater, 2600, interval=40, blit=False)
    # fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', lambda event: on_key(event))

    plt.xlim(-5,5)
    # plt.ylim(3, 20)
    ax.set_zlim(-3, 8)

    plt.show()
    pass

if __name__ == '__main__':
    arg = get_argument()
    data_set = load_data_from_folder(arg.path)
    print("Shape of data_set;", data_set.shape)
    # print(data_set)
    # data_set = get_wqds(arg.file)

    plot_acc_frames(data_set, arg.num_of_frames, arg.vx, arg.vy, arg.vz)
    pass