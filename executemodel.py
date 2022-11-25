import sys
import numpy as np
from sklearn.metrics import accuracy_score
from hmm import DHMM
from fileprocessing import read_annotations_from_file
from fileprocessing import load_epochs_from_file
from featureextraction import convert_features_to_codebook
from featureextraction import convert_to_frequency_domain
import warnings
import time

# Main
# run: python src/executemodel.py data/SC4002E0-PSG.edf data/SC4002EC-Hypnogram.edf
# ====
if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    """ First Step is the handling of the commandline arguments. Two arguments are expected: The *PSG.edf files are whole-night polysmnographic sleep recordings containing EEG
    and the *Hypnogram.edf which contain the annotations of the sleep patterns that correspond to the PSG.
    """
    print(len(sys.argv))
    if len(sys.argv) == 3:
        start = time.time()
        data_file = sys.argv[1]
        annotations_file = sys.argv[2]

        sleep_stages_dictionary = {'Sleep stage W': 5, 'Sleep stage 1': 3, 'Sleep stage 2': 2, 'Sleep stage 3': 1,
                                   'Sleep stage 4': 0, 'Sleep stage R': 4, 'Movement time': 6}
        sleep_stages = read_annotations_from_file(annotations_file, sleep_stages_dictionary)
        print(sleep_stages)
        print(len(np.where(sleep_stages == sleep_stages_dictionary['Sleep stage 1'])[0]))
        print(len(np.where(sleep_stages == sleep_stages_dictionary['Sleep stage 4'])[0]))
        num_states = len(np.unique(sleep_stages))
        # annotations contain long sequences of the awake state at the beginning and the end and those are removed
        actual_sleep_epochs_indices = np.where(sleep_stages != sleep_stages_dictionary['Sleep stage W'])
        sleep_start_index = actual_sleep_epochs_indices[0][0]
        sleep_end_index = actual_sleep_epochs_indices[0][-1] + 1

        sleep_stages = sleep_stages[sleep_start_index:sleep_end_index]
        # EEG epoching is when specific time-windows are extracted from the continuous EEG signal
        # The long duration of fixed length epochs of 30 seconds is selected for achieving better frequency resolution for the epoch_length variable below
        # The EOG and EEG signals in the data were each sampled at 100 Hz and thus fs is 100
        epochs = load_epochs_from_file(data_file, epoch_length=30, fs=100)


        epochs = epochs[sleep_start_index:sleep_end_index, :]
        # this function takes the eeg data and creates the eeg features
        # feature extraction in the frequency domain ( or convert a signal into its frequency components)
        features = convert_to_frequency_domain(epochs, epoch_length=30, fs=100)

        num_observations = 20  # number of discrete features groups
        codebook, epoch_codes = convert_features_to_codebook(features, num_observations)

        training_percentage = 0.8  # % of data used for training the model
        nr_epochs = sleep_stages.shape[0]
        sleep_stages_train, sleep_stages_test = np.split(sleep_stages, [int(training_percentage * nr_epochs)])
        epoch_codes_train, epoch_codes_test = np.split(epoch_codes, [int(training_percentage * nr_epochs)])

        hmm = DHMM(num_states, num_observations)
        hmm.train(sleep_stages_train, epoch_codes_train)
        # Viterbi Algorithm is used to calculate the states in the DHMM
        x = hmm.get_state_sequence(epoch_codes_test)

        sleep_stages_reverse = {y: x for x, y in sleep_stages_dictionary.items()}
        actual_phases = list(map(lambda phase: sleep_stages_reverse[phase], sleep_stages_test))
        predicted_phases = list(map(lambda phase: sleep_stages_reverse[phase], x))
        print("Actual sleep phases when paired with predicted sleep phases:")

        array = [0, 0, 0, 0, 0, 0, 0]
        for actual, predicted in zip(actual_phases, predicted_phases):
            print("Actual:",actual, "Predicted:", predicted)
            if (actual != predicted):
                    array[sleep_stages_dictionary[actual]] += 1
        print("Model Accuracy:", accuracy_score(sleep_stages_test, x))
        for i in range(len(array)):
            print(sleep_stages_reverse[i], "missed: ", array[i])
        end = time.time()
        print("Total time to execute the model: ", end-start)
    else:
        print('\n'.join(["Unexpected number of commandline arguments.",
                         "Usage: ./executemodel.py {data_file} {annotation_file}"]))
