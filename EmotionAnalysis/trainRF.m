%% -- Train and Optimize Random Forest Model -- %%
load('tfidfVariables.mat');
%% -- First find the optimal number of trees -- %%
% WARNING, these loops can take hours, change the range to increment by and
% up to different amounts. For the poster image I used 1:2:100
numberTreesRange = 1:2:100;
crossValErrors = zeros(numel(numberTreesRange), 1);
surrogateCrossValErrors = zeros(numel(numberTreesRange), 1);
% Loop through different min leaf sizes and compute cross-validated errors
% with the gdi split
for i = 1:numel(numberTreesRange)
    numberTrees = numberTreesRange(i);
    disp(numberTrees);
    Mdl = fitcensemble(X_Train, Y_Train, 'Method', 'Bag' ,'NumLearningCycles', numberTrees, 'CrossVal', 'on');
    crossValErrors(i) = kfoldLoss(Mdl);
end
% Surrogate on
t = templateTree('Surrogate','on');
for i = 1:numel(numberTreesRange)
    numberTrees = numberTreesRange(i);
    disp(numberTrees);
    Mdl = fitcensemble(X_Train, Y_Train, 'Method', 'Bag' ,'NumLearningCycles', numberTrees, 'Learners', t, 'CrossVal', 'on');
    surrogateCrossValErrors(i) = kfoldLoss(Mdl);
end

% Compare the models with different numbers of trees and 
% with and without surrogate. Specifically compare the cross validation
% error of the models.
figure;
plot(numberTreesRange, crossValErrors, 'ro-', 'LineWidth', 2, 'MarkerSize', 4);
hold on;
plot(numberTreesRange, surrogateCrossValErrors, 'bo-', 'LineWidth', 2, 'MarkerSize', 4);
xlabel('Num Trees');
ylabel('Cross-Validated Error');
title('Cross-Validation Error vs. Number of trees');
grid on;

%% Find best min leaf size and split criterion with our optimal number of trees
% Use the optimal number of trees calculated from the code above
[minValue, minIndex] = min(crossValErrors);
optimalTrees = minIndex * 2; 
% The optimal number of trees is around 43
% Specify the range of min leaf sizes to explore
% As above the variable can be changed to any desired increment and ending
% value. The image in the poster used the 1:2:100
minLeafSizeRange = 1:2:100;
numMinLeafSizes = numel(minLeafSizeRange);
GDIcrossValErrors = zeros(numMinLeafSizes, 1);
twoing_crossValErrors = zeros(numMinLeafSizes, 1);
deviance_crossValErrors = zeros(numMinLeafSizes, 1);

% Loop through different min leaf sizes and compute cross-validated errors
% with the gdi split
% Use template trees to change the DT values of the RF
for i = 1:numMinLeafSizes
    minLeafSize = minLeafSizeRange(i);
    disp(minLeafSize);
    t = templateTree('SplitCriterion', 'gdi', "MinLeafSize", minLeafSize);
    Mdl = fitcensemble(X_Train, Y_Train, 'Method', 'Bag' ,'NumLearningCycles', optimalTrees,'Learners', t, 'CrossVal', 'on');
    GDIcrossValErrors(i) = kfoldLoss(Mdl);
end

% Do the same with twoing split
for i = 1:numMinLeafSizes
    minLeafSize = minLeafSizeRange(i);
    disp(minLeafSize);
    t = templateTree('SplitCriterion', 'twoing', "MinLeafSize", minLeafSize);
    Mdl = fitcensemble(X_Train, Y_Train, 'Method', 'Bag' ,'NumLearningCycles', optimalTrees,'Learners', t, 'CrossVal', 'on');
    twoing_crossValErrors(i) = kfoldLoss(Mdl);
end

% And with deviance split
for i = 1:numMinLeafSizes
    minLeafSize = minLeafSizeRange(i);
    disp(minLeafSize);
    t = templateTree('SplitCriterion', 'deviance', "MinLeafSize", minLeafSize);
    Mdl = fitcensemble(X_Train, Y_Train, 'Method', 'Bag' ,'NumLearningCycles', optimalTrees,'Learners', t, 'CrossVal', 'on');
    deviance_crossValErrors(i) = kfoldLoss(Mdl);
end

%% Plot the results of the different splits and minleafsize
% Compare the different models by cross-validation-error
figure;
plot(minLeafSizeRange, GDIcrossValErrors, 'ro-', 'LineWidth', 2, 'MarkerSize', 4);
hold on;
plot(minLeafSizeRange, twoing_crossValErrors, 'bo-', 'LineWidth', 2, 'MarkerSize', 4);
plot(minLeafSizeRange, deviance_crossValErrors, 'go--', 'LineWidth', 2, 'MarkerSize', 4);
xlabel('Min Leaf Size');
ylabel('Cross-Validated Error');
title('Cross-Validation Error vs. Min Leaf Size');
grid on;
%% Train and export the optimized Random Forest model
% We weren't able to learn anything from the optimization of the min leaf
% size, the default value of 1 (default) and using GDI Split Criterion (default)
% works best, so no template tree is needed
optimalTrees = 43; % Remove this if running the loops from above
tic % time the training of the RF
RF = fitcensemble(X_Train, Y_Train, 'Method', 'Bag' ,'NumLearningCycles', optimalTrees);
toc
save('trainedRF.mat', 'RF');