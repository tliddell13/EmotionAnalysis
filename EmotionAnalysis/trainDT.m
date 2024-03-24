%% -- Train and Optimize Decision Tree Model -- %%
load('tfidfVariables.mat');
% Specify the range of min leaf sizes to explore
% This value can be changed to search by desired increments. The image on
% the poster was created using 1:2:100
minLeafSizeRange = 1:2:100;

numMinLeafSizes = numel(minLeafSizeRange);
GDIcrossValErrors = zeros(numMinLeafSizes, 1);
twoing_crossValErrors = zeros(numMinLeafSizes, 1);
deviance_crossValErrors = zeros(numMinLeafSizes, 1);

% Loop through different min leaf sizes and compute cross-validated errors
% with the gdi split
for i = 1:numMinLeafSizes
    minLeafSize = minLeafSizeRange(i);
    Mdl = fitctree(X_Train, Y_Train, 'MinLeafSize', minLeafSize, 'CrossVal', 'on', 'SplitCriterion','gdi');
    GDIcrossValErrors(i) = kfoldLoss(Mdl);
end

% Do the same with twoing split
for i = 1:numMinLeafSizes
    minLeafSize = minLeafSizeRange(i);
    Mdl = fitctree(X_Train, Y_Train, 'MinLeafSize', minLeafSize, 'CrossVal', 'on', 'SplitCriterion','twoing');
    twoing_crossValErrors(i) = kfoldLoss(Mdl);
end

% And with deviance split
for i = 1:numMinLeafSizes
    minLeafSize = minLeafSizeRange(i);
    Mdl = fitctree(X_Train, Y_Train, 'MinLeafSize', minLeafSize, 'CrossVal', 'on', 'SplitCriterion','deviance');
    deviance_crossValErrors(i) = kfoldLoss(Mdl);
end

%% Plot the results
% This plot compares the decision tree performance with different split
% criterion and increasing minleafsize
figure;
plot(minLeafSizeRange, GDIcrossValErrors, 'ro-', 'LineWidth', 2, 'MarkerSize', 4);
hold on;
plot(minLeafSizeRange, twoing_crossValErrors, 'bo-', 'LineWidth', 2, 'MarkerSize', 4);
plot(minLeafSizeRange, deviance_crossValErrors, 'go--', 'LineWidth', 2, 'MarkerSize', 4);
xlabel('Min Leaf Size');
ylabel('Cross-Validated Error');
title('Cross-Validation Error vs. Min Leaf Size');
grid on;
%% Train and export the decision tree
% Increasing the number of minimum leaf size quickly brings down
% performance. Best performance found at minleafsize of 6 and
% splitCriterion using GDI (defualt)
tic % time the training of the decision tree 
DTree = fitctree(X_Train, Y_Train, 'MinLeafSize', 6);
toc
% Export the trained model
save('trainedDTree.mat', 'DTree');


