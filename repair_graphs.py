import os
import argparse
import h5py
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir",type=str,default="./s3dis/graphs",help="Directory of the graphs")
    args = parser.parse_args()
    n = 0
    for file in os.listdir(args.dir):
        file_dir = args.dir + "/" + file
        hf = h5py.File(file_dir, "r")
        uni_senders = np.array(hf["uni_senders"], copy=True) 
        n_sps = np.array(hf["n_sps"], copy=True)
        hf.close()
        if n_sps == uni_senders.shape[0]:
            continue
        print("Will delete scene {0}".format(file))
        os.remove(file_dir)
        n+=1
    print("{0} graphs deleted".format(n))

if __name__ == "__main__":
    main()