import random
import json
import tqdm

def load_data(fname):
    data = json.load(open(fname))
    print(len(data))
    return data



def write_data(data, fname):
    with open(fname, 'w') as f:
        json.dump(data, f, indent=4)




