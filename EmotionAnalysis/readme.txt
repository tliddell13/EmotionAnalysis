# Files / description
analysis.m - This file corresponds with the preprocessing and initial analysis section of my poster.
DTvsRF.m - This file is used for evaluating the models on the final test set.
methodology.m - If you are curious about intermediate results or material not included in the poster, refer here.
tfidf.m - This file is used to turn the data into tfidf matrices and create a variable of the data.
trainDT.m - Contains the grid search and hyperparameter optimization for the decision tree model, as well as figures associated with this phase found in the Parameter Selection and Experimental Results section of the poster. 
trainRF.m - Contains the grid search and hyperparameter optimization for the random forest model, as well as figures associated with this phase found in the Parameter Selection and Experimental Results section of the poster.

# Variables / description
trainedDT.mat - contains the pretrained decision tree model.
trainedRF.mat - contains the pretrained random forest model.
tfidf.mat - this contains the preprocessed test and training variables (split into X and Y) 

# Data
The dataset is split into three files.
- An unsplit dataset for analysis emotion_dataset.csv
- Training data (%70) training_data.csv
- Testing data (%30) testing_data.csv

# Test the models
To run a test on the models use file DTvsRF.m. This file will load the tfidf variables for the testing data, make predictions, and create confusion matrices.

# Train the models
To retrain the models from scratch use the model's corresponding file trainDT.m or trainRF.m and adjust grid searches and hyperparameters as desired. This file will load the tfidf variables for the training data and use cross-validation to evaluate metrics.

# Requirements
"Matlab"                                                                               "9.14.0.2206163 (R2023a)"
"Text Analytics Toolbox"                                                               "1.10"       true      "TA"                  
"Statistics and Machine Learning Toolbox"                                              "12.5"       true      "ST"  