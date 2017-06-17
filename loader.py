import numpy as np
import os
import struct
import csv

FMT_CPCB = '<qffff'
LEN_CPCB = 24

def load_cepton_csv(fn):
    data = np.loadtxt(fn, delimiter=',')
    # print("CSV-DATA: ", data)
    return data

def load_cpcd(fn):
    datas = []
    with open(fn, 'rb') as bfile:
        while True:
            data = bfile.read(24)
            if len(data) < LEN_CPCB:
                break
            datas.append(struct.unpack(FMT_CPCB, data))
            # print("timestamp, X/Y/Z/Intensity", data[0], data[1], data[2], data[3], data[4])
    ds = np.array(datas)
    # print("Shape of data from file: ", ds.shape, fn)
    return ds

def load_file_with_id(dir, id, format='cpcd'):
    print("loading data from folder...")

    for _, _, files in sorted(os.walk(dir)):
        # print("number of files: ", len(files))
        # for file in files:
        #     file_path = os.path.join(dir, file)
        #     # print('file name:', file_path)
        #     # data_set = np.concatenate((data_set,load_cpcd(file_path)))
        #     data_set.append(load_cpcd(file_path))
        file_loader = load_cpcd
        if 'csv' == format:
            file_loader = load_cepton_csv
        # data = load_cpcd(dir + files[id])
        print("Loading file: ", dir + files[id])
        data = file_loader(dir + files[id])
        # print(data)
        break

    return np.array(data)

def load_data_from_folder(dir, format = 'cpcd', max_file = 1000):
    # data_set = load_cpcd(FILE_NAME)
    print("loading data from folder...")
    data_loader = load_cpcd
    if 'csv' == format:
        data_loader = load_cepton_csv

    data_set = []
    for _, _, files in ( os.walk(dir) ):
        f = 0
        for file in sorted(files):
            if(f > max_file):
                break
            file_path = os.path.join(dir, file)
            print('file name:', file_path)
            # data_set = np.concatenate((data_set,load_cpcd(file_path)))
            data_set.append(data_loader(file_path))
            f += 1

    print("FRAME-ID: ", len(data_set))
    return np.array(data_set)