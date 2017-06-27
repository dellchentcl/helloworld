#!/home/dell0/anaconda3/envs/tf/bin/python

from loader import *
from param import *

if __name__ == '__main__':
    arg = get_argument()
    i = arg.file
    o = arg.ofile
    convert(i,o)
    pass