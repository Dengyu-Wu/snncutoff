import matplotlib.pyplot as plt
import numpy as np
import os
import torch 

import struct
import os

V3 = "aedat3"
V2 = "aedat"  # current 32bit file format
V1 = "dat"  # old format

EVT_DVS = 0  # DVS event type
EVT_APS = 1  # APS event


def loadaerdat(datafile='/tmp/aerout.dat', length=0, version=V2, debug=1, camera='DVS128'):
    """    
    load AER data file and parse these properties of AE events:
    - timestamps (in us), 
    - x,y-position [0..127]
    - polarity (0/1)
    @param datafile - path to the file to read
    @param length - how many bytes(B) should be read; default 0=whole file
    @param version - which file format version is used: "aedat" = v2, "dat" = v1 (old)
    @param debug - 0 = silent, 1 (default) = print summary, >=2 = print all debug
    @param camera='DVS128' or 'DAVIS240'
    @return (ts, xpos, ypos, pol) 4-tuple of lists containing data of all events;
    """
    # constants
    aeLen = 8  # 1 AE event takes 8 bytes
    readMode = '>II'  # struct.unpack(), 2x ulong, 4B+4B
    td = 0.000001  # timestep is 1us   
    if(camera == 'DVS128'):
        xmask = 0x00fe
        xshift = 1
        ymask = 0x7f00
        yshift = 8
        pmask = 0x1
        pshift = 0
    elif(camera == 'DAVIS240'):  # values take from scripts/matlab/getDVS*.m
        xmask = 0x003ff000
        xshift = 12
        ymask = 0x7fc00000
        yshift = 22
        pmask = 0x800
        pshift = 11
        eventtypeshift = 31
    else:
        raise ValueError("Unsupported camera: %s" % (camera))

    if (version == V1):
        #print ("using the old .dat format")
        aeLen = 6
        readMode = '>HI'  # ushot, ulong = 2B+4B

    aerdatafh = open(datafile, 'rb')
    k = 0  # line number
    p = 0  # pointer, position on bytes
    statinfo = os.stat(datafile)
    if length == 0:
        length = statinfo.st_size    
    #print ("file size", length)
    
    # header
    lt = aerdatafh.readline()
    while lt and lt[0] == "#":
        p += len(lt)
        k += 1
        lt = aerdatafh.readline() 
        if debug >= 2:
            print (str(lt))
        continue
    
    # variables to parse
    timestamps = []
    xaddr = []
    yaddr = []
    pol = []
    
    # read data-part of file
    aerdatafh.seek(p)
    s = aerdatafh.read(aeLen)
    p += aeLen
    
    #print (xmask, xshift, ymask, yshift, pmask, pshift)    
    while p < length:
        addr, ts = struct.unpack(readMode, s)
        # parse event type
        if(camera == 'DAVIS240'):
            eventtype = (addr >> eventtypeshift)
        else:  # DVS128
            eventtype = EVT_DVS
        
        # parse event's data
        if(eventtype == EVT_DVS):  # this is a DVS event
            x_addr = (addr & xmask) >> xshift
            y_addr = (addr & ymask) >> yshift
            a_pol = (addr & pmask) >> pshift


            #if debug >= 3: 
            #    print("ts->", ts)  # ok
            #    print("x-> ", x_addr)
            #    print("y-> ", y_addr)
            #    print("pol->", a_pol)

            timestamps.append(ts)
            xaddr.append(x_addr)
            yaddr.append(y_addr)
            pol.append(a_pol)
                  
        aerdatafh.seek(p)
        s = aerdatafh.read(aeLen)
        p += aeLen        

    #if debug > 0:
    #    try:
    #        #print ("read %i (~ %.2fM) AE events, duration= %.2fs" % (len(timestamps), len(timestamps) / float(10 ** 6), (timestamps[-1] - timestamps[0]) * td))
    #        n = 5
    #        print ("showing first %i:" % (n))
    #        print ("timestamps: %s \nX-addr: %s\nY-addr: %s\npolarity: %s" % (timestamps[0:n], xaddr[0:n], yaddr[0:n], pol[0:n]))
    #    except:
    #        print ("failed to print statistics")
    txyp = {
        't': [],
        'x': [],
        'y': [],
        'p': []
    }    
    timestamps = np.asarray(timestamps)
    start = int(np.argwhere(timestamps == 0)[0])
    txyp['x'] = np.asarray(xaddr[start:])
    txyp['y'] = np.asarray(yaddr[start:])
    txyp['t'] = np.asarray(timestamps[start:])
    txyp['p'] = np.asarray(pol[start:])            
    return txyp


file_num = 0
y_train = []
x_train = []
y_test = []
x_test = []
class_names = []
random_mode = 0 # 1 or 0
train_filename = '/LOCAL/dengyu/dvs_dataset/dvs-cifar10-1-1s/train/{}.pt'
test_filename = '/LOCAL/dengyu/dvs_dataset/dvs-cifar10-1-1s/test/{}.pt'

for root,dirs,files in os.walk('/opt/LOCAL/share/datasets/cifar10_dvs'):
    root = str(root)
    num = 0
    for names in dirs:
        class_names.append(names)
        
key_train = -1
key_test = -1
for root,dirs,files in os.walk('/opt/LOCAL/share/datasets/cifar10_dvs'):
    root = str(root)
    _num = 0
    sample_num = len(files)
    sample_array = np.arange(sample_num)
    sample_for_training = int(sample_num*0.9)
    #np.random.shuffle(sample_array)     #shuffle train and test
    for names in files:
        names = str(names)
        if names.endswith(".aedat"):
            path = root+'/'+names 
            if sample_array[_num] < sample_for_training:
                _train = True
                key_train += 1
            else:
                _train = False
                key_test += 1                
            _num += 1
            file_num +=1 
            print('\r file number: ',file_num,end='')
            #if file_num>4:
            #    continue
            data = loadaerdat(path)
            
            N = 1
            split_num = 1
            H = 128
            W = 128
            t_start = 0.1e6
            t_tot = t_start+split_num*1.1e6
            data_t = data['t']-data['t'][0]
            loc_tot = np.where(np.logical_and(data_t > t_start, data_t < t_tot))
            data_t = data_t[loc_tot[0]]-data_t[loc_tot[0]][0]
            slot = (data_t[-1])/split_num

            if _train:
                for n in range(split_num):
                    __p = n + 1
                    for _n in range (N):
                        _p=_n+1
                        #_frame = np.zeros(shape=[128,128,N*2])
                        frame = np.zeros(shape=[2, H * W])
                        loc = np.where(np.logical_and(data_t < __p*slot-(N-_p)/N*slot, 
                                                          data_t >= (__p-1)*slot+_n/N*slot))
                        x = data['x'][loc[0]].astype(int)  # avoid overflow
                        y = data['y'][loc[0]].astype(int)
                        p = data['p'][loc[0]]
                        mask = []
                        mask.append(p == 0)
                        mask.append(np.logical_not(mask[0]))
                        for c in range(2):
                            position = y[mask[c]] * W + x[mask[c]]
                            events_number_per_pos = np.bincount(position)
                            frame[c][np.arange(events_number_per_pos.size)] += events_number_per_pos

                        if _n == 0:
                            frame = frame.reshape((2, H, W))
                            frame = np.expand_dims(frame,axis=0)
                            _frame = frame
                        else:
                            frame = frame.reshape((2, H, W))
                            frame = np.expand_dims(frame,axis=0)
                            _frame = np.append(_frame,frame,0)
                    #x_train.append(_frame)
                    
                    num=0
                    for subname in class_names:
                        _path = path.split("/")
                        if subname == _path[-2]:
                            break
                        num += 1                
                    torch.save([torch.Tensor(_frame), torch.Tensor([num,])],train_filename.format(key_train))
            else:
                for n in range(split_num):
                    __p = n + 1
                    for _n in range (N):
                        _p=_n+1
                        #_frame = np.zeros(shape=[128,128,N*2])
                        frame = np.zeros(shape=[2, H * W])
                        loc = np.where(np.logical_and(data_t < __p*slot-(N-_p)/N*slot, 
                                                          data_t >= (__p-1)*slot+_n/N*slot))

                        x = data['x'][loc[0]].astype(int)  # avoid overflow
                        y = data['y'][loc[0]].astype(int)
                        p = data['p'][loc[0]]
                        mask = []
                        mask.append(p == 0)
                        mask.append(np.logical_not(mask[0]))
                        for c in range(2):
                            position = y[mask[c]] * W + x[mask[c]]
                            events_number_per_pos = np.bincount(position)
                            frame[c][np.arange(events_number_per_pos.size)] += events_number_per_pos
                        if _n == 0:
                            frame = frame.reshape((2, H, W))
                            frame = np.expand_dims(frame,axis=0)
                            _frame = frame
                        else:
                            frame = frame.reshape((2, H, W))
                            frame = np.expand_dims(frame,axis=0)
                            _frame = np.append(_frame,frame,0)

                    #x_test.append(_frame)
                    num=0
                    
                    for subname in class_names:
                        _path = path.split("/")
                        if subname == _path[-2]:
                            break
                        num += 1      
                    torch.save([torch.Tensor(_frame), torch.Tensor([num,])],test_filename.format(key_test))
