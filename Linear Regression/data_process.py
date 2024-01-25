import os
import pandas as pds

def read_data_file():
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    data_file_path = os.path.realpath(curr_dir+'/data/player-data.csv')
    print('Reading data file...')
    data_csv = pds.read_csv(data_file_path)
    return data_csv