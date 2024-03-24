%% -- Decisision Tree (DT) vs. Random Forest (RF) for Emotion Analysis -- %%
% load the training and testing data
load('tfidfVariables.mat');

%% -- Decision Tree -- %%
% load the trained decision tree
load('trainedDTree.mat');
% calculate the training accuracy
DTtrainingAccuracy = 1 - resubLoss(DTree);
% make predictions on the testing data using the decision tree
% and measure time it takes
tic
predictions = predict(DTree, X_Test);
toc
% create a confusion matrix from the predictions
confMatrix = confusionmat(Y_Test, predictions);
% Accuracy of DT
DTaccuracy = sum(diag(confMatrix)) / sum(confMatrix(:));
% Precision, Recall, and F1 Score for each class
DTprecision = zeros(1, 6);
DTrecall = zeros(1, 6);
DTf1Score = zeros(1, 6);
for i = 1:6
    DTprecision(i) = confMatrix(i, i) / sum(confMatrix(:, i));
    DTrecall(i) = confMatrix(i, i) / sum(confMatrix(i, :));
    DTf1Score(i) = 2 * (DTprecision(i) * DTrecall(i)) / (DTprecision(i) + DTrecall(i));
end
% Calculate Precision, Recall, and F1 Score using a macro average
DTprecision = mean(DTprecision);
DTrecall = mean(DTrecall);
DTf1Score = mean(DTf1Score);
% Display the results
disp('Training Accuracy:');
disp(DTtrainingAccuracy);
disp('Accuracy:');
disp(DTaccuracy);
disp('Precision:');
disp(DTprecision);
disp('Recall:');
disp(DTrecall);
disp('F1 Score:');
disp(DTf1Score);

figure;
labels = {'sadness', 'joy', 'love', 'anger', 'fear', 'surprise'};
confusionchart(confMatrix, labels, 'RowSummary','row-normalized', 'ColumnSummary', 'column-normalized');
title('Decision Tree Emotion Classification');

%% -- Random Forest -- %%
% Repeat the same process as above for the Random Forest
load('trainedRF.mat');
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

figure;
confusionchart(confMatrix, labels, 'RowSummary','row-normalized', 'ColumnSummary', 'column-normalized');
title('Random Forest Emotion Classification');

%% -- Table comparing results -- %%
% Create a table
metricsTable = table(...
    [DTtrainingAccuracy; RFtrainingAccuracy], ...
    [DTaccuracy; RFaccuracy], ...
    [DTprecision; RFprecision], ...
    [DTrecall; RFrecall], ...
    [DTf1Score; RFf1Score], ...
    'VariableNames', {'TrainingAccuracy', 'Accuracy', 'Precision', 'Recall', 'F1Score'}, ...
    'RowNames', {'DecisionTree', 'RandomForest'});

% Display the table
disp('Comparison of Decision Tree and Random Forest Metrics:');
disp(metricsTable);



