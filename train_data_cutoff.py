import os
import re
import csv
import glob
import pickle
import string
import collections

import numpy as np
import gensim

from io import open


training_matrix_dir = './data/Holmes_Training_Data_5gram_matrix/small'
training_matrix_dir_new = './data/Holmes_Training_Data_5gram_matrix/5000'

#big_file_dir = './data/Holmes_Training_Data_5gram_matrix/big'

input_files = glob.glob(os.path.join(training_matrix_dir, "*.TXT"))

line_limit = 5000

def check(input_files):
    for input_file in input_files:
        print('Processing file: ' + input_file)
        temp = []
        with open(input_file, "r", errors='ignore') as f:
            # sentences = f.readlines()
            output = open(training_matrix_dir_new+"/"+os.path.basename(input_file), "w")
            for i, line in enumerate(f):
                if (i >= line_limit):
                    break
                else:
                    output.write(line)
            output.close()
            # os.rename(input_file, big_file_dir+'/'+os.path.basename(input_file))
            # if (i+1 < line_limit):
            #     os.rename(input_file, training_matrix_dir_new+'/'+os.path.basename(input_file))
check(input_files)