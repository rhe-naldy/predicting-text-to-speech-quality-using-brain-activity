# machine-learning-predicting-text-to-speech-quality-using-brain-activity

[Predicting Text-To-Speech Quality using Brain Activity](https://ieeexplore.ieee.org/document/9975857) is a group research paper made for Machine Learning course in the fourth semester, which was published by [The 2022 International Conference on Internet of Things and Intelligence System (IoTaIS 2022)](http://iotais.org/) and posted to the IEEE Xplore digital library. The paper can be viewed by clicking [here](https://ieeexplore.ieee.org/document/9975857). Below is the abstract to the paper.

# ABSTRACT
The perceived audio quality is one of the key factors that may determine a text-to-speech system’s success in the market. Therefore, it is important to conduct audio
quality evaluation before releasing such system into the market. Evaluating the synthesized audio quality is usually done either subjectively or objectively with their own advantages and disadvantages. Subjective methods usually require a large amount of time and resources, while objective methods lack human influence factors, which are crucial for deriving the subjective perception of quality. These human influence factors are manifested inside an individual’s brain in forms such as
electroencephalograph (EEG). Thus, in this study, we performed audio quality prediction using EEG data. Since the data used in this study is small, we also compared the prediction result of the augmented and the non-augmented data. Our result shows that certain method yield significantly better prediction with augmented training data.

### Code Explanation
1. [featureExtraction](https://github.com/rhe-naldy/machine-learning-predicting-text-to-speech-quality-using-brain-activity/blob/main/featureExtraction.py)

In this file, we performed fast Fourier transform (FFT) in order to extract spectrogram from each EEG sample with a 1-second window and a 0.5 second overlap, followed by normalization with the highest frequency limited to 45 Hz.

2. [augmentedFeatureExtraction](https://github.com/rhe-naldy/machine-learning-predicting-text-to-speech-quality-using-brain-activity/blob/main/augmentedFeatureExtraction.py)

In this file, we augmented the raw EEG signal 30 times using Gaussian noise. Then, we performed fast Fourier transform (FFT) in order to extract spectrogram from each EEG sample with a 1-second window and a 0.5 second overlap, followed by normalization with the highest frequency limited to 45 Hz.

3. [peakPicking.py](https://github.com/rhe-naldy/machine-learning-predicting-text-to-speech-quality-using-brain-activity/blob/main/peakPicking.py)

In this file, we performed peak picking technique with a minimal horizontal distance of 18 in samples between neighboring peaks, followed by signal resampling and standardization of data.

4. [nonAugmentedCV.py](https://github.com/rhe-naldy/machine-learning-predicting-text-to-speech-quality-using-brain-activity/blob/main/nonAugmentedCV.py)

In this file, we constructed three machine learning models, consisting of SVR, MLP, and Decision Tree Regressor. We trained and tested the models independently using nested cross-validation by dividing the dataset into 4 different sets of data. Then, we tested every possible combination of train-test data with the train data using 3 sets of non augmented data and the remaining one set of non augmented data as the test data.

5. [augmentedCV.py](https://github.com/rhe-naldy/machine-learning-predicting-text-to-speech-quality-using-brain-activity/blob/main/augmentedCV.py)

In this file, we constructed three machine learning models, consisting of SVR, MLP, and Decision Tree Regressor. We trained and tested the models independently using nested cross-validation by dividing the dataset into 4 different sets of data. Then, we tested every possible combination of train-test data with the train data using 3 sets of augmented data and the remaining one set of non augmented data as the test data.

### Dataset
The dataset used in this study is thhe PhySyQX dataset, which contains sample audio stimuli, twelve subjective dimensions collected from a total of 21 participants for each audio stimulus, and electroencephalography (EEG) and functional near-infrared spectroscopy (fNIRS) records of each participant when the audio stimuli were presented. In this study, we only utilized the EEG record and the average of overall impression opinion scores.

### Methods
In this study, we applied Peak Picking Technique and EEG Spectrogram Extraction as preprocessing techniques. Meanwhile, we also augmented the dataset using different Gaussian noise deviation in order to evaluate whether the regression methods used in the study could yield better results. Support Vector Machine (SVM), decision tree, and neural network are the regression methods used in this study. All of te regression metods used in this study is implemented using the Scikit-learn package. The neural network model consist of two hidden layers each with 128 units. The decision tree model is implemented with the parameters set to the default settings. The SVM model used the Radial Basis Function (RBF) kernel, with the regularization set to five. 

### Results
In comparing the results yielded by the regression methods, we performed significance test using Wilcoxon signed-rank test with the following configuration (α = 0.01, T = 42, N = 21).

![image](https://user-images.githubusercontent.com/45966986/215100143-963786c5-ca36-4e50-b7f7-aa0d69c1a2f2.png)

Table I shows the average MOS regression RMSE of non-augmented EEG features between Peak Picking Technique and EEG Spectrogram Extraction. The result shows that the SVM model performed significantly better compared to the other methods, while the decision tree model yielded better results compared to the neural network model.

![image](https://user-images.githubusercontent.com/45966986/215101375-6f950c08-74f0-43e6-81f9-2c45f31b19a5.png)

Table II shows the average MOS regression RMSE between non-augmented and augmented EEG features using EEG spectrogram extraction as the preprocessing technique. The result shows that the SVM model performed significantly better compared to the other methods in both non-augmented and augmented setting. Additionally, the performance of the SVM model is not affected by the augmentation of EEG features, with the same results yielded between using non-augmented and augmented EEG features.

The SVM model performed significantly better compared to the other methods could be due to the audio stimuli represented is easily separated, which is also reflected from the MOS score distribution of audio samples, which can be seen below.

![image](https://user-images.githubusercontent.com/45966986/215103851-d2cfb6c0-a538-4c79-a174-58942d324836.png)

Meanwhile, the difference in prediction score of neural network between implementing Peak Picking Technique and EEG Spectrogram Extraction could be due to the higher dimension of the spectrogram compared to the data generated from implementing Peak Picking Technique. Thus, it can be concluded that the difference 
