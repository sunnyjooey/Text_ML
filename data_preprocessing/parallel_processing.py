# Parallel processing for parsing SVO triples

import pandas as pd
import pickle

import process
import feature
import parse
import time
import multiprocessing
from threading import Lock


print ("number of cores: ", multiprocessing.cpu_count())

CPU_COUNT = multiprocessing.cpu_count()
NUM_CHUNKS = 100

class Counter(object):
    def __init__(self):
        self.value = 0
        self.lock = Lock()

    def incr(self, x):
        with self.lock:
            self.value += 1
            print(str(self.value) + " / " + str(NUM_CHUNKS) + " Finsished")

def work(dataTuples):
    index = dataTuples[0]
    agChunk = dataTuples[1]
    print("Running chunk: ", index)
    parsed = parse.parse_svo(agChunk, 'tokenized_text_agg')
    with open('data/parsed_' + str(index) + '.pkl', 'wb') as f:
        pickle.dump(parsed, f)
    
    print("Finished chunk: ", index)


if __name__ == '__main__':
    start = time.time()
    with open('data/aggregated.pkl', 'rb') as f:
        ag = pickle.load(f)

    print("loaded: ", time.time() - start)

    length = ag.shape[0]
    chunckSize = length / NUM_CHUNKS


    dataTuples = []

    for index in range(0, NUM_CHUNKS):
        startIndex = chunckSize * index
        endIndex = chunckSize * (index + 1)
        agChunk = ag.loc[startIndex:endIndex,:]
        dataTuples.append((index, agChunk))

    counter = Counter()

    pool = multiprocessing.Pool(processes=CPU_COUNT)
    # pool.apply_async(work, dataTuples, callback=counter.incr)
    r = [pool.apply_async(work, (x,), callback=counter.incr) for x in dataTuples]
    # pool.map(work, dataTuples)
    pool.close()
    pool.join()

    print("done: ", time.time() - start)
