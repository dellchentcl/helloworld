import struct
import numpy as np
from param import *

FMT_CPCB = '<qffff'
LEN_CPCB = 24

def load_cpcd(fn):
    datas = []
    with open(fn, 'rb') as bfile:
        while True:
            data = bfile.read(LEN_CPCB)
            if len(data) < LEN_CPCB:
                break
            datas.append(struct.unpack(FMT_CPCB, data))
            # print("timestamp, X/Y/Z/Intensity", data[0], data[1], data[2], data[3], data[4])

    return np.array(datas)

FMT_FRAME_HDR = '<iiiii'
FMT_FH_LEN = 20
frame_description_size = 76

def get_wqds(fn):
    ds = []
    
    with open(fn, 'rb') as bf:
        pos = 0
        tmp = bf.read(4)
        pos += 4

        frame_num = struct.unpack('<i', tmp)
        frame_num = frame_num[0]
        # print("FRAME NUM: ", frame_num)

        null_data = bf.read(4) # first_hdr_off
        pos += 4

        '''TODO: need read according to first_hdr_off '''
        fhdrs = []
        for i in range(frame_num):
            data = bf.read(FMT_FH_LEN)
            pos += FMT_FH_LEN
            tmp = struct.unpack(FMT_FRAME_HDR, data)
            fhdrs.append(tmp)
            # print("FRAME %i, OFFSET: %i, LENGTH: %i" % (i, tmp[0], tmp[1]))

        frame_start = fhdrs[0][0]
        # print("FRAME_START AT: ", frame_start)

        if(frame_start > pos):
            null_data = bf.read(frame_start - pos )
            pos = frame_start

        frame = []
        ''' read a frame '''
        for i in range(frame_num):
            ''' frame description '''
            sync = bf.read(4)
            pos += 4
            # print("SYNC: ", sync)

            speed = bf.read(4)
            pos += 4
            
            timestamp = bf.read(16)
            pos += 16
            
            channel_num = bf.read(4)
            pos += 4
            
            train_head_off = bf.read(4)
            pos += 4
            
            train_dir = bf.read(4)
            pos += 4
            
            train_tail_num = bf.read(4)
            pos += 4

            train_tail_off = bf.read(4)
            pos += 4
            
            train_tail_dir = bf.read(4)
            pos += 4
            
            measure_bias = bf.read(4)
            pos += 4
            
            location_valid = bf.read(4)
            pos += 4
            
            system_time = bf.read(16)
            pos += 16
            
            null_data = bf.read(4)  # redundant speed
            pos += 4

            # frame_description_size = pos - frame_start
            # print("DESCRIPTOR SIZE: ", frame_description_size)
            
            pts_to_read = fhdrs[i][1] - frame_description_size # frame len
            # print("POINTS_TO_READ: ", pts_to_read)
            print("FRAME: ", i)
            for j in range(int(pts_to_read/LEN_CPCB)):
                tmp = bf.read(LEN_CPCB)
                pos += LEN_CPCB
                if len(tmp) < LEN_CPCB:
                    break
                rec = struct.unpack(FMT_CPCB, tmp)
                print(rec)
                frame.append(rec)

            ds.append(np.array(frame))
            # print("FRAME-DATA: \n", frame)

            if i == frame_num-1:
                break

            frame_start = fhdrs[i+1][0]
            if frame_start > pos:
                null_data = bf.read(frame_start - pos)
            pos = frame_start

    ds = np.array(ds)
    # print("SHAPE OF DATASET: ", ds.shape)

    return ds

if __name__ == '__main__':
    arg = get_argument()
    ds = get_wqds(arg.file)
    # print(ds)
    pass