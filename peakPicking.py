import scipy.io as sio
import scipy
from scipy import signal
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
from joblib import Parallel, delayed
import multiprocessing
import pandas as pd
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor as MLP
from sklearn.tree import DecisionTreeRegressor as DTR

max = 0
peaks = [] #peaks[subject(0-20)][audio(0-43)][eegChannel(0-63)][0-255]

for x in range(8, 29):
    scaler = StandardScaler()
    source_folder = 'Research-Materials/EEG/'
    target_folder = 'Research-Materials/EEG-Spectrogram/'
    
    data = sio.loadmat(source_folder + 'sub' + str(x) + '_noCSD.mat')
    
    indexes = [i[0][0][0] for i in data["EEG"][0][0][25][0] if i[0][0][0] in np.arange(5, 49)]
    records = data["EEG"][0][0][15].transpose(2,0,1)[1:-1]
    
    #loop per audio
    #loop per eeg channels
    #append per eeg channels
    #append per audio
    #
    #peak picking
    
    audio = []
    
    for i in range(0, 44):
        channel = []
        for j in range(0, 64):
            pick, _ = scipy.signal.find_peaks(records[i][j], distance = 18)
            pick = scipy.signal.resample(pick, 256)
            channel.append(pick)
        
        channel = scaler.fit_transform(channel)
        audio.append(channel)
    
    peaks.append(audio)
'''    
#load data terlebih dahulu
scores = pd.read_excel("Research-Materials/scores.xlsx")
#MOS
Q = scores["'Quality'"]
y = []

for i in range(0, 44):
    index = i
    temp = 0
    temp_index = 0;
    for j in range(8, 29):
        temp += Q[i + temp_index]
        temp_index += 44

    temp = temp / 21
    y.append(temp)
    #print("MOS Stimuli " + str(i + 5) + ": " + str(temp))

cv_index = [[5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45],
            [6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46],
            [7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47],
            [8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48]]

#model per subject
def generate_model(index, methods):
    if methods == 1:
        model = SVR(kernel = 'rbf', C = 5)
    elif methods == 2:
        model = MLP(hidden_layer_sizes=(128, 128,))
    elif methods == 3:
        model = DTR()
    
    rmse = []
    #test per fold
    for j in range(0, 4):
        x_train = []
        x_test = []
        y_train = []
        y_test = []
        
        #train data
        for k in range(0, 4):
            if k == j:
                continue;
            
            for l in range(0, 11):
                y_train.append(y[cv_index[k][l] - 5])
                x_train.append(peaks[index][cv_index[k][l] - 5])

        #test data
        for l in range(0, 11):
            y_test.append(y[cv_index[j][l] - 5])
            x_test.append(peaks[index][cv_index[j][l] - 5])
        
        x_train = np.reshape(x_train, (33, 16384))
        x_test = np.reshape(x_test, (11, 16384))
        
        model.fit(x_train, y_train)
        
        y_pred = model.predict(x_test)
        
        temp_rmse = mean_squared_error(y_test, y_pred, squared = False)
        
        rmse.append(temp_rmse)
        
        
    mean = (rmse[0] + rmse[1] + rmse[2] + rmse[3]) / 4
    return round(mean, 3)

for i in range(1, 4):
    if i == 1:
        print("SVR")
    elif i == 2:
        print("MLP")
    elif i == 3:
        print("DTR")
    
    model_generation = Parallel(n_jobs = multiprocessing.cpu_count())(delayed(generate_model)(index, i) for index in range(0, 21))    
    
    avg = 0
    for j in range(8, 29):
        avg += model_generation[j - 8]
        print("Subject " + str(j) + ": " + str(model_generation[j - 8]))

    avg = round(avg / 21, 3)
    print("Avg: " + str(avg))
'''