%%
% In this file the data is turned into tfidf matrices. To understand the
% reasoning behind the preprocessing carried out in this file, refer to the
% TFIDF section of the methodology.m file
% Load the testing and training data from csv
% The data is split 70 - 30
dataTrain = readtable('training_data.csv');
dataTest = readtable('testing_data.csv');
% tokenize the text
documents = tokenizedDocument(dataTrain.text);
% remove the stop words
documents = removeStopWords(documents);
bag = bagOfWords(documents);
bag = removeInfrequentWords(bag, 2);
% Use tfidf to turn the data into a matrix
X_Train = full(tfidf(bag, tokenizedDocument(dataTrain.text)));
Y_Train = dataTrain.label;
% Test set
% Do the same as above
X_Test = full(tfidf(bag, tokenizedDocument(dataTest.text)));
Y_Test = dataTest.label;
% We can find which features are most important using a decision tree
DTree = fitctree(X_Train, Y_Train);
featureImportance = predictorImportance(DTree);
nonZeroIndices = find(featureImportance ~= 0);
X_Train = X_Train(:, nonZeroIndices);
X_Test = X_Test(:,nonZeroIndices);
% Save the training and testing data as mat variables
save('tfidfVariables.mat', 'X_Train', 'X_Test', 'Y_Train', 'Y_Test');




