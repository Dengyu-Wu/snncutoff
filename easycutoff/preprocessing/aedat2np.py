import matplotlib.pyplot as plt
import numpy as np
import os

def load_aedat_v3(file_name: str):
    import struct

    '''
    :param file_name: path of the aedat v3 file
    :type file_name: str
    :return: a dict whose keys are ``['t', 'x', 'y', 'p']`` and values are ``numpy.ndarray``
    :rtype: Dict
    This function is written by referring to https://gitlab.com/inivation/dv/dv-python . It can be used for DVS128 Gesture.
    '''
    with open(file_name, 'rb') as bin_f:
        # skip ascii header
        line = bin_f.readline()
        while line.startswith(b'#'):
            if line == b'#!END-HEADER\r\n':
                break
            else:
                line = bin_f.readline()

        txyp = {
            't': [],
            'x': [],
            'y': [],
            'p': []
        }
        while True:
            header = bin_f.read(28)
            if not header or len(header) == 0:
                break

            # read header
            e_type = struct.unpack('H', header[0:2])[0]
            e_source = struct.unpack('H', header[2:4])[0]
            e_size = struct.unpack('I', header[4:8])[0]
            e_offset = struct.unpack('I', header[8:12])[0]
            e_tsoverflow = struct.unpack('I', header[12:16])[0]
            e_capacity = struct.unpack('I', header[16:20])[0]
            e_number = struct.unpack('I', header[20:24])[0]
            e_valid = struct.unpack('I', header[24:28])[0]

            data_length = e_capacity * e_size
            data = bin_f.read(data_length)
            counter = 0

            if e_type == 1:
                while data[counter:counter + e_size]:
                    aer_data = struct.unpack('I', data[counter:counter + 4])[0]
                    timestamp = struct.unpack('I', data[counter + 4:counter + 8])[0] | e_tsoverflow << 31
                    x = (aer_data >> 17) & 0x00007FFF
                    y = (aer_data >> 2) & 0x00007FFF
                    pol = (aer_data >> 1) & 0x00000001
                    counter = counter + e_size
                    txyp['x'].append(x)
                    txyp['y'].append(y)
                    txyp['t'].append(timestamp)
                    txyp['p'].append(pol)
            else:
                # non-polarity event packet, not implemented
                pass
        txyp['x'] = np.asarray(txyp['x'])
        txyp['y'] = np.asarray(txyp['y'])
        txyp['t'] = np.asarray(txyp['t'])
        txyp['p'] = np.asarray(txyp['p'])
        return txyp

    
def split_aedat_files_to_np(fname: str, aedat_file: str, csv_file: str, output_dir: str):
    events = load_aedat_v3(aedat_file)
    print(f'Start to split [{aedat_file}] to samples.')
    # read csv file and get time stamp and label of each sample
    # then split the origin data to samples
    csv_data = np.loadtxt(csv_file, dtype=np.uint32, delimiter=',', skiprows=1)

    # Note that there are some files that many samples have the same label, e.g., user26_fluorescent_labels.csv
    label_file_num = [0] * 11

    # There are some wrong time stamp in this dataset, e.g., in user22_led_labels.csv, ``endTime_usec`` of the class 9 is
    # larger than ``startTime_usec`` of the class 10. So, the following codes, which are used in old version of SpikingJelly,
    # are replaced by new codes.


    for i in range(csv_data.shape[0]):
        # the label of DVS128 Gesture is 1, 2, ..., 11. We set 0 as the first label, rather than 1
        label = csv_data[i][0] - 1
        t_start = csv_data[i][1]
        t_end = csv_data[i][2]
        #t_end  = t_start+1.5*1e6
        mask = np.logical_and(events['t'] >= t_start, events['t'] < t_end)
        file_name = os.path.join(output_dir, f'{fname}_{label}.npz')
        
        np.savez(file_name,
                 t=events['t'][mask],
                 x=events['x'][mask],
                 y=events['y'][mask],
                 p=events['p'][mask]
                 )
        print(f'[{file_name}] saved.')
        label_file_num[label] += 1



if __name__ == '__main__':
    data = []
    labels = []
    aedat_list = []
    path_to_dataset = '/LOCAL/dengyu/dvs_dataset/DvsGesture'
    
    path_to_save = '/LOCAL/dengyu/dvs_dataset/GestureNP/'
    
    for root,dirs,files in os.walk(path_to_dataset):
        root = str(root)
        for names in files:
            names = str(names)
            path = root+'/'+names
            if names.endswith(".aedat"):
                aedat_file = names
                csv_file = names[:-6] +'_labels.csv'

                if float(names[4:6]) >= 24:
                    fname = 'test_'+names[:-6]
                else:
                    fname = 'train_'+names[:-6]

                split_aedat_files_to_np(fname,root+'/'+aedat_file,root+'/'+csv_file,path_to_save)  
