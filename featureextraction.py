import numpy as np
from sklearn import preprocessing
import random

def convert_to_frequency_domain(epochs, fs, epoch_length):
    print("Extracting features in the frequency domain...")
    N = epochs.shape[0]
    L = epoch_length * fs # signal length
    f = np.linspace(0, L-1, L) * fs/L
    delta1, delta2, theta1, theta2, alpha1, alpha2, beta1, beta2 = 0, 4, 4, 8, 8, 13, 13, 30
    # setting up the frequency intervals for the different sleep stages
    all_indices = np.where((f <= beta2))
    delta_indices = np.where((f >= delta1) & (f <= delta2))
    theta_indices = np.where((f >= theta1) & (f <= theta2))
    alpha_indices = np.where((f >= alpha1) & (f <= alpha2))
    beta_indices = np.where((f >= beta1) & (f <= beta2))
    nr_features = 6 # number of features to be calculated
    features = np.zeros((N, nr_features))
    # calculation of delta, theta, alpha and beta band power ratios
    for index in range(N):
        epoch = epochs[index, :]
        # just takes the absolute value of the number, removes the negative sign if any
        # and np.fft.fft helps in frequency-domain (spectral) representation of the signal by
        # transforming the  signal from the time domain to the frequency domain
        Y = abs(np.fft.fft(epoch))
        mean_total_power = np.mean(Y[all_indices])
        features[index,:] = (mean_total_power, np.mean(f[all_indices] * Y[all_indices]) / mean_total_power,np.mean(Y[delta_indices]) / mean_total_power, np.mean(Y[theta_indices]) / mean_total_power,
        np.mean(Y[alpha_indices]) / mean_total_power, np.mean(Y[beta_indices]) / mean_total_power)
    return preprocessing.scale(features)

def convert_features_to_codebook(features, num_observations):
    print("Discretizing features...")
    # number of records
    N = features.shape[0]
    nr_features = features.shape[1]
    # vector quantization and codebook generation
    minimums = np.min(features, 0)
    maximums = np.max(features, 0)
    #Convert zip to list
    min_max = list(zip(minimums,maximums))
    # initialize random codebook using random.uniform which creates a uniform distribution
    codebook = np.array([[random.uniform(min_max[column][0], min_max[column][1])
        for column in range(nr_features)] for row in range(num_observations)])
    epoch_codes = np.zeros(N, dtype=np.int)
    epoch_codes_prev = np.zeros(N, dtype=np.int)
    count = 0
    while True:
        count += 1
        for index_epoch in range(N):
            distances = np.zeros(num_observations)
            for index_codebook in range(num_observations):
                #https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html
                distances[index_codebook] = np.linalg.norm(codebook[index_codebook, :] - features[index_epoch, :])
              #get the indices of the minimum values along an axis
            epoch_codes[index_epoch] = np.argmin(distances)
        if np.array_equal(epoch_codes_prev, epoch_codes):
            break
        epoch_codes_prev = np.copy(epoch_codes)
        # calculating the new center points by taking the mean
        for code in np.unique(epoch_codes):
            code_indices = np.where(epoch_codes == code)
            grouped_vectors = np.squeeze(features[code_indices, :])
            codebook[code] = np.mean(grouped_vectors, axis=0)
    print("Count ",count)
    return codebook, epoch_codes