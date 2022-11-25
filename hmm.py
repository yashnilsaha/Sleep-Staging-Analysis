import numpy as np
from collections import Counter


class DHMM:
    # Discrete Hidden Markov Model
    def __init__(self, nr_states, nr_groups):
        # finite set of possible states
        self.nr_states = nr_states
        # finite set of possible observations
        self.nr_observations = nr_groups
        # initial state distribution
        self.pi = np.zeros(nr_states)
        # transition probabilities or state transition matrix
        self.A = np.zeros((self.nr_states, self.nr_states))
        # observation or emission probabilities or probability distribution when at a given state
        self.B = np.zeros((self.nr_states, self.nr_observations))

    def train(self, sleep_stages_train, epoch_codes_train):
        unique, counts = np.unique(sleep_stages_train, return_counts=True)
        nr_epochs = sleep_stages_train.shape[0]
        self.pi = np.array(counts) / nr_epochs

        self.A = np.zeros((self.nr_states, self.nr_states))
        for (x,y), c in Counter(zip(sleep_stages_train, sleep_stages_train[1:])).items():
            self.A[x][y] = c
        # self.A = np.random.randint(100, size=(self.nr_states, self.nr_states))
        self.A = self.A / self.A.sum(axis=1, keepdims=True)

        self.B = np.zeros((self.nr_states, self.nr_observations))
        for (x,y),c in Counter(zip(sleep_stages_train, epoch_codes_train)).items():
            self.B[x,y] = c
        self.B = self.B / self.B.sum(axis=1, keepdims=True)

    def get_state_sequence(self, observ):
        # What sequence of states best explains the sequence of observations is done by the Viterbi algorithm
        T = observ.shape[0]
        delta = np.zeros((T, self.nr_states))
        psi = np.zeros((T, self.nr_states))
        delta[0] = np.log(self.pi) + np.log(self.B[:, observ[0]])
        for t in range(1,T):
            for j in range(self.nr_states):
                delta[t,j] = np.max(delta[t-1] + np.log(self.A[:,j])) + np.log(self.B[j, observ[t]])
                psi[t,j] = np.argmax(delta[t-1] + np.log(self.A[:,j]))
        states = np.zeros(T, dtype=int)
        states[T-1] = np.argmax(delta[T-1])
        for t in range(T-2, -1, -1):
            states[t] = psi[t+1, states[t+1]]
        return states