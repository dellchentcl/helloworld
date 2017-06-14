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
    return parser.parse_args()