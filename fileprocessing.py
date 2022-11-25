import numpy as np
import pyedflib

def read_annotations_from_file(filename, sleep_stages_dict):
    print("Reading annotations of the sleep patterns that correspond to the PSGs...")
    f = pyedflib.EdfReader(filename)   
    annotations = f.readAnnotations()
    sleep_stages = []
    for index in range(np.shape(annotations)[1] - 1): # last stage is "Sleep stage ?" - filler
        sleep_stages = sleep_stages + [sleep_stages_dict[annotations[2][index]]]*int((int(annotations[1][index]) / 30))
    sleep_stages = np.array(sleep_stages)
    return sleep_stages

def load_epochs_from_file(filename, epoch_length, fs):
    print("Loading epochs...")
    f = pyedflib.EdfReader(filename)
    # https://pyedflib.readthedocs.io/en/latest/ref/edfreader.html#pyedflib.EdfReader.readSignal
    # Returns the physical data of signal chn into sigbuf
    sigbuf = f.readSignal(0)
    cols = epoch_length * fs # signal length
    epochs = np.reshape(sigbuf, (-1, cols))
    return epochs