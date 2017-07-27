import os
import re
import csv
import glob
import pickle
import string
import collections

import numpy as np
from io import open

data_dir = './data/Holmes_Training_Data/'
output_dir = './data/Holmes_Training_Data_Clean/'

input_files = glob.glob(os.path.join(data_dir, "*.TXT"))
print(input_files)

def clean_str(string):
    """
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data
    """
    string = re.sub(r"\[([^\]]+)\]", " ", string)
    string = re.sub(r"\(([^\)]+)\)", " ", string)
    string = re.sub(r"[^A-Za-z0-9,!?.;]", " ", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r";", " ; ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def preprocess(input_files, ngram_size=5):
    #i = 0

    for input_file in input_files:
        with open(input_file, "r", encoding='latin1', errors='ignore') as f:
            print "Reading from " + input_file
            data = f.read()
            
            # text cleaning or make them lower case, etc.
            data = clean_str(data) 
            
            sentences = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=[.?;!])\s",data)
            
            ngram_dir = './data/Holmes_Training_Data_' + str(ngram_size) + 'gram'
            
            if not os.path.exists(ngram_dir):
                os.makedirs(ngram_dir)
            
            for index, sent in enumerate(sentences):
                s = re.sub(r"\.", " . ", sent)
                s = re.sub(r"\s{2,}", " ", s)
                tokens = s.strip().split()
                
                if len(tokens) >= ngram_size:
                    ngram_arr = zip(*[tokens[i:] for i in range(ngram_size)])
                    for index2, ngram in enumerate(ngram_arr):
                        f = open(ngram_dir + '/' + os.path.basename(input_file), 'a')
                        if (index == len(sentences)-1 and index2 == len(ngram_arr)-1):
                            f.write(' '.join(list(ngram)))
                        else:
                            f.write(' '.join(list(ngram))+'\n')
            f.close()

                    
    # self.vocab, self.words = self.build_vocab(x_sent)
    # self.vocab_size = len(self.words)

    # with open(vocab_file, 'wb') as f:
    #     pickle.dump(self.words, f)
    
    
    # """
    # print("Vocabulary snippet:")
    # print(self.words[:100])
    # """

    # self.data = self.map_id(x_sent)
    # # Save the data to data.npy
    # np.save(data_file, self.data)

    # """
    # print("Text snippet:")
    # print(x_sent[:10])
    
    # print("Text-to-id snippet:")
    # print(self.data[:10])
    # """
    # self.num_data = len(self.data)
    # self.num_batches = self.num_data // self.batch_size

preprocess(input_files)