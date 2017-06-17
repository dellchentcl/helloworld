import argparse

def get_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=int, default=15, help="Frame ID to be showed")
    parser.add_argument("--path", type=str, default="./", help="Point cloud data directory")
    parser.add_argument('--file', type=str, default='no_exist.csv', help='file name in specific format')
    parser.add_argument('--format', type=str, default='cpcd', help='file format')
    parser.add_argument('--gap', type=float, default=0, help='merge duplicate data in center')
    parser.add_argument('--vx', type=float, default=0, help='speed in x axis')
    parser.add_argument('--vy', type=float, default=0, help='speed in y axis')
    parser.add_argument('--vz', type=float, default=0, help='speed in z axis')
    parser.add_argument('--sy', type=float, default=0, help='start point of y for showing')
    parser.add_argument('--ey', type=float, default=0, help='end point of y for showing')
    parser.add_argument('--num_of_frames', type=int, default=5, help='Number of latest frames for accumulate')
    parser.add_argument('--opcl', type=str, default='tmp.pcl', help='The file name for output pcl file')
    parser.add_argument('--files', type=str, default='[tmp.pcl]', help='identifi several files for display')

    return parser.parse_args()