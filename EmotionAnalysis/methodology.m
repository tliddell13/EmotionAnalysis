%% ---- METHODOLOGY AND INTERMEDIATE RESULTS ---- %%
% This file is dedicated to my preliminary results particularly regarding
% why I applied the preprocessing and feature removal I did

% First I attempted to use mean embedding with extremely poor results on default RF
% and DT models

%% -- MEAN EMBEDDING FAILURE -- %%
% Load the data from the CSV
data = readtable('training_data.csv');
documents = tokenizedDocument(data.text);
documents = removeStopWords(documents);
% Partition the data 70 - 30
cvp = cvpartition(data.label,'HoldOut',0.3);
% Train a custom word embedding
emb = trainWordEmbedding(documents, 'Model','cbow');
words = emb.Vocabulary;
V = word2vec(emb,words);
XY = tsne(V);
textscatter(XY,words)
% This loop calculates a mean embedding for each message
% Matlab tutorial: 
% ehttps://uk.mathworks.com/help/textanalytics/ug/classify-documents-using-document-embeddings.html
meanEmbedding = zeros(numel(documents),emb.Dimension);
for k=1:numel(documents)
    words = string(documents(k));
    wordVectors = word2vec(emb,words);
    wordVectors = rmmissing(wordVectors);
    % A few sentences contain words that don't appear. Set these to a
    % vector of zero
    if ~isempty(wordVectors)
        meanEmbedding(k, :) = mean(wordVectors, 1);
    else 
        meanEmbedding(k, :) = zeros(1, emb.Dimension);
    end
end

X_Train = meanEmbedding(training(cvp),:);
X_Test = meanEmbedding(test(cvp),:);
Y_Train = data.label(training(cvp), :);
Y_Test = data.label(test(cvp), :);
model = fitcensemble(X_Train, Y_Train, 'Method', 'Bag');
predictions = predict(model, X_Test);
confMatrix = confusionmat(Y_Test, predictions);
RFaccuracy = sum(diag(confMatrix)) / sum(confMatrix(:));
RFprecision = zeros(1, 6);
RFrecall = zeros(1, 6);
RFf1Score = zeros(1, 6);
% Precision, recall, and f1score for all 6 emotions
for i = 1:6
    RFprecision(i) = confMatrix(i, i) / sum(confMatrix(:, i));
    RFrecall(i) = confMatrix(i, i) / sum(confMatrix(i, :));
    RFf1Score(i) = 2 * (RFprecision(i) * RFrecall(i)) / (RFprecision(i) + RFrecall(i));
end
% Use mean to calculate the macro averages 
% A minority class will have an equal weight on the score by doing this.
RFprecision = mean(RFprecision);
RFrecall = mean(RFrecall);
RFf1Score = mean(RFf1Score);
disp('Accuracy:');
disp(RFaccuracy);
disp('Precision:');
disp(RFprecision);
disp('Recall:');
disp(RFrecall);
disp('F1 Score:');
disp(RFf1Score);


% Do the same as above, but use the fast text word 
% embedding for the bag of words 
ftemb = fastTextWordEmbedding;
% Performs slightly better using the pretrained word embedding, but still
% not great
meanEmbedding = zeros(numel(documents),ftemb.Dimension);
for k=1:numel(documents)
    words = string(documents(k));
    wordVectors = word2vec(ftemb,words);
    wordVectors = rmmissing(wordVectors);
    meanEmbedding(k,:) = mean(wordVectors,1);
end
X_Train = meanEmbedding(training(cvp),:);
X_Test = meanEmbedding(test(cvp),:);
Y_Train = data.label(training(cvp), :);
Y_Test = data.label(test(cvp), :);
model = fitcensemble(X_Train, Y_Train, 'Method', 'Bag');
predictions = predict(model, X_Test);
% Same calculations as above, I should of changed this to a function as I
% used the same code several times
confMatrix = confusionmat(Y_Test, predictions);
RFaccuracy = sum(diag(confMatrix)) / sum(confMatrix(:));
RFprecision = zeros(1, 6);
RFrecall = zeros(1, 6);
RFf1Score = zeros(1, 6);
for i = 1:6
    RFprecision(i) = confMatrix(i, i) / sum(confMatrix(:, i));
    RFrecall(i) = confMatrix(i, i) / sum(confMatrix(i, :));
    RFf1Score(i) = 2 * (RFprecision(i) * RFrecall(i)) / (RFprecision(i) + RFrecall(i));
end
RFprecision = mean(RFprecision);
RFrecall = mean(RFrecall);
RFf1Score = mean(RFf1Score);
disp('Accuracy:');
disp(RFaccuracy);
disp('Precision:');
disp(RFprecision);
disp('Recall:');
disp(RFrecall);
disp('F1 Score:');
disp(RFf1Score);
%% -- TFIDF METHODOLOGY -- %%
% Load the testing and training data from csv
% The data is split 70 - 30
dataTrain = readtable('training_data.csv');
dataTest = readtable('testing_data.csv');
% Training set
% Create a bag of words with the training data
% There is no noticeable advantage to including the test data in the bag of
% words
% tokenize the text
documents = tokenizedDocument(dataTrain.text);
% remove the stop words
% the stop words don't improve the models performance but add more features
% which increases computation time
documents = removeStopWords(documents);

bag = bagOfWords(documents);
bagInfrequent = removeInfrequentWords(bag, 2);
% Use tfidf to turn the data into a matrix
X_Train = full(tfidf(bagInfrequent, tokenizedDocument(dataTrain.text)));
Y_Train = dataTrain.label;

% Test set
% Do the same as above
X_Test = full(tfidf(bagInfrequent, tokenizedDocument(dataTest.text)));
Y_Test = dataTest.label;
% We can find which features are most important using a decision tree
DTree = fitctree(X_Train, Y_Train);
featureImportance = predictorImportance(DTree);
nonZeroIndices = find(featureImportance ~= 0);
% By doing this we are able to reduce the number of features in our
% data from 5108 to 765
filteredX_Train = X_Train(:, nonZeroIndices);
filteredX_Test = X_Test(:,nonZeroIndices);

DTree1 = fitctree(filteredX_Train, Y_Train, 'PredictorSelection', 'curvature');
predictions = predict(DTree1, filteredX_Test);
confMat = confusionmat(Y_Test, predictions);
accuracy = sum(diag(confMat)) / sum(confMat(:));
precision = confMat(2, 2) / sum(confMat(:, 2));
recall = confMat(2, 2) / sum(confMat(2, :));
disp('Preprocessing + no DT unimportant features');
disp(['Accuracy: ' num2str(accuracy)]);
disp(['Precision: ' num2str(precision)]);
disp(['Recall: ' num2str(recall)]);

% We can also see the effect that leaving the stop words and short words in
% has on this methodology for filtering 
% repeat the process above without removing stop words or infrequent words
documents = tokenizedDocument(dataTrain.text);
bag = bagOfWords(documents);
documents = tokenizedDocument(dataTrain.text);
X_Train = full(tfidf(bag, tokenizedDocument(dataTrain.text)));
Y_Train = dataTrain.label;
X_Test = full(tfidf(bag, tokenizedDocument(dataTest.text)));
Y_Test = dataTest.label;
DTree = fitctree(X_Train, Y_Train, 'PredictorSelection', 'curvature');
featureImportance = predictorImportance(DTree);
nonZeroIndices = find(featureImportance ~= 0);
filteredX_Train = X_Train(:, nonZeroIndices);
filteredX_Test = X_Test(:,nonZeroIndices);
DTree2 = fitctree(filteredX_Train, Y_Train);
predictions = predict(DTree2, filteredX_Test);
confMat = confusionmat(Y_Test, predictions);
accuracy = sum(diag(confMat)) / sum(confMat(:));
precision = confMat(2, 2) / sum(confMat(:, 2));
recall = confMat(2, 2) / sum(confMat(2, :));
disp('No Preprocessing + no DT unimportant features');
disp(['Accuracy: ' num2str(accuracy)]);
disp(['Precision: ' num2str(precision)]);
disp(['Recall: ' num2str(recall)]);

% Just for good measure a decision tree without removing any features
% how long does it take?
tic
DTree3 = fitctree(X_Train, Y_Train);
toc
predictions = predict(DTree3, X_Test);
confMat = confusionmat(Y_Test, predictions);
accuracy = sum(diag(confMat)) / sum(confMat(:));
precision = confMat(2, 2) / sum(confMat(:, 2));
recall = confMat(2, 2) / sum(confMat(2, :));
disp('No Feature Removal');
disp(['Accuracy: ' num2str(accuracy)]);
disp(['Precision: ' num2str(precision)]);
disp(['Recall: ' num2str(recall)]);

% How long does it take to train a RF without feature removal?
tic
RF = fitcensemble(X_Train, Y_Train, "Method","Bag");
toc
predictions = predict(RF, X_Test);
confMat = confusionmat(Y_Test, predictions);
accuracy = sum(diag(confMat)) / sum(confMat(:));
precision = confMat(2, 2) / sum(confMat(:, 2));
recall = confMat(2, 2) / sum(confMat(2, :));
disp('No Feature Removal');
disp(['Accuracy: ' num2str(accuracy)]);
disp(['Precision: ' num2str(precision)]);
disp(['Recall: ' num2str(recall)]);
% The results are very similar so I decided to use the preprocessing and
% removal of unimportant features which brought the number of features down
% significantly 
%% BIG RANDOM FOREST
% this one is slow and doesn't offer any performance advantages
tic % how long does it take to train?
RF = fitcensemble(X_Train, Y_Train, "Method","Bag", "NumLearningCycles",700);
toc
% Same calculations as above
RFtrainingAccuracy = 1 - resubLoss(RF);
tic
predictions = predict(RF, X_Test);
toc
confMatrix = confusionmat(Y_Test, predictions);
RFaccuracy = sum(diag(confMatrix)) / sum(confMatrix(:));
RFprecision = zeros(1, 6);
RFrecall = zeros(1, 6);
RFf1Score = zeros(1, 6);
for i = 1:6
    RFprecision(i) = confMatrix(i, i) / sum(confMatrix(:, i));
    RFrecall(i) = confMatrix(i, i) / sum(confMatrix(i, :));
    RFf1Score(i) = 2 * (RFprecision(i) * RFrecall(i)) / (RFprecision(i) + RFrecall(i));
end
RFprecision = mean(RFprecision);
RFrecall = mean(RFrecall);
RFf1Score = mean(RFf1Score);
disp('Training Accuracy:');
disp(RFtrainingAccuracy);
disp('Accuracy:');
disp(RFaccuracy);
disp('Precision:');
disp(RFprecision);
disp('Recall:');
disp(RFrecall);
disp('F1 Score:');
disp(RFf1Score);
labels = {'sadness', 'joy', 'love', 'anger', 'fear', 'surprise'};
figure;
confusionchart(confMatrix, labels, 'RowSummary','row-normalized', 'ColumnSummary', 'column-normalized');
title('Random Forest Emotion Classification');
