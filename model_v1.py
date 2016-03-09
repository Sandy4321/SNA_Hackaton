import os
from os.path import isfile, join

import pandas as pd
import numpy as np


## initalizing parquet

import parquet
import csv
import StringIO

class Options(object):

    def __init__(self, col=None, format='csv', no_headers=True, limit=-1):
        self.col = col
        self.format = format
        self.no_headers = no_headers
        self.limit = limit


def read_parquet_data (filename):
    
    actual_raw_data = StringIO.StringIO()
    parquet.dump(filename, Options(format='csv'), out=actual_raw_data)
    actual_raw_data.seek(0, 0)
    actual_data = list(csv.reader(actual_raw_data, delimiter='\t'))
    
    return actual_data


## read graph data

import time
import gzip
from multiprocessing import Pool, Lock


GRAPH_DIR  = "/home/sirorezka/python_proj/SNA_Hackaton/Data/trainGraph"
N_PROCS = 4




all_graph_data = []
# aprx 8 minutes
def read_graph_files (gz_file):
    
    graph_data = []
    i_counter = 0
    with gzip.open(os.path.join(GRAPH_DIR,gz_file), 'rb') as f:
    	#print gz_file
        for line in f:
            i_counter += 1
            data = line.split("\t")
            user = data[0]
            all_friends = line.split("\t")[1].replace("{(","").replace(")}", "").replace("\n","").split("),(")
            all_friends = map (lambda x: [user] + x.split(","),all_friends)
            graph_data += all_friends
            
            if i_counter % 1000 == 0:
            	print  gz_file, " ",i_counter
    
    graph_data = pd.DataFrame(graph_data, dtype="int32")
    print "finished"
    return graph_data


def read_graph_files_par ():
    

        tic = time.time()
        gz_files = [f for f in os.listdir(GRAPH_DIR) if f.endswith('.gz')]
        #print (gz_files[0:2])
        pool = Pool(processes = N_PROCS)
        graph_data_full = pool.map_async(read_graph_files, gz_files)
        print "finished_all"
        pool.close()
        pool.join()  


        toc = time.time() - tic
        print "elapsed time", toc
        
        return graph_data_full
        #all_graph_data = pd.DataFrame(all_graph_data)
        





f1 = 'part-v008-o000-r-00014.gz'
f2 = 'part-v008-o000-r-00002.gz'

gr_data = read_graph_files_par ()
# tic = time.time()
# gr_data = read_graph_files(f1)
# gr_data = read_graph_files(f2)

# toc = time.time() - tic


print ("\n")
print (gr_data[0:100])
        