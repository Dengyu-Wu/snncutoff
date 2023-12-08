
import matplotlib.pyplot as plt
import numpy as np
import os
import torch 

path_to_dataset = '/LOCAL/dengyu/dvs_dataset/GestureNP'   #Path to the directory of dataset
split_ratio = 0.9   # 90% samples for training

N = 16                  # Frame number of single sample
split_num = 1          # Split one sample into samples
H = 128     
W = 128
t_start = 0.0e6        # start time 
t_tot = t_start+1.3e6*split_num  # compute total time of spike train

file_num = 0
y_train = []
x_train = []
y_test = []
x_test = []
key_train = -1
key_test = -1

visualise_num_file = 100

for root,dirs,files in os.walk(path_to_dataset):
    root = str(root)
    for names in files:
        names = str(names)
        path = root+'/'+names    
        if names[0:5] == 'train':
            _train = True
            key_train += 1
        else:
            _train = False
            key_test += 1
        file_num +=1 
        print('\rfile number: ',file_num,end='')
        if file_num == visualise_num_file:
            if names[-6:-4] == '10':
                label = int(names[-6:-4])
            else:
                label = int(names[-5:-4])

            data = np.load(path)

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

            break
frame = _frame

import matplotlib.pyplot as plt 
frame = frame/np.max(frame)



import numpy as np
import matplotlib.pyplot as plt


fig = plt.figure(figsize=(8, 8))
columns = 4
rows = 4
for i in range(1, columns*rows +1):
    img = frame[i-1,0]
    fig.add_subplot(rows, columns, i)
    plt.imshow(img)
plt.show()