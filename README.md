# Sleep-Staging-Analysis
Automating Human Sleep Staging Analysis from Polysomnographic Recordings with a Discrete Hidden Markov Predictive Model

#Data
Go to https://www.physionet.org/content/sleep-edfx/1.0.0/ and download the polysomnographic data files.

#Running
executemodel.py must be run with two arguments. The first argument to the program must the name of an individual data file (named with PSG.edf) and the second must be the corresponding annotations file with the same file name (named with Hypnogram.edf).

#Requirements
This model requires the following modules:
*numpy
*sklearn
*pyedflib

#Troubleshooting
*For certain data files, the model will run into an error such as not being able to read the annotations file or having a state sequence error.
