%Dataset analysis
%% Load the entire dataset
data = readtable('emotion_dataset.csv');
dataY = data.label;
%% Make a pie chart of the different emotions
labels = {'sadness: ', 'joy: ', 'love: ', 'anger: ', 'fear: ', 'surprise: '};

C = categorical(dataY,[0 1 2 3 4 5]);
N = histcounts(C);
pie = pie(N);
pText = findobj(pie,'Type','text');
percentValues = get(pText,'String');  
combinedtxt = strcat(labels.',percentValues); 
pText(1).String = combinedtxt(1);
pText(2).String = combinedtxt(2);
pText(3).String = combinedtxt(3);
pText(4).String = combinedtxt(4);
pText(5).String = combinedtxt(5);
pText(6).String = combinedtxt(6);
pie

%% Create a bag of words and find which words are most important  
% for prediction using a decision tree
docs = data.text;
docs = tokenizedDocument(docs);
docs = removeStopWords(docs);
docs = removeShortWords(docs, 2);
bag = bagOfWords(docs);
% Most frequent words in the entire dataset
frequentWords = topkwords(bag,10);
bag = removeInfrequentWords(bag, 2);
figure;
bar(categorical(frequentWords.Word), frequentWords.Count);
title('Top Ten Words');
xlabel('Word');
ylabel('Count');
X = full(tfidf(bag, tokenizedDocument(data.text)));
Y = data.label;
DTree = fitctree(X, Y, 'PredictorSelection', 'curvature');
featureImportance = predictorImportance(DTree);
nonZeroIndices = find(featureImportance ~= 0);
[sortedValues, sortedIndices] = sort(featureImportance, 'descend');
top10Indices = sortedIndices(1:10);
top10Importance = sortedValues(1:10);
importantWords = bag.Vocabulary(:,top10Indices);

%% Make a bar chart of the most important words in the vocabulary
figure;
bar = bar(categorical(importantWords), top10Importance);
title('Top Words');
xlabel('Word');
ylabel('Count');

% Analyse the words contained in each labels vocabulary
%% Sad Vocab
sad = data.text(data.label == 0);
sadDocs = tokenizedDocument(sad);
sadDocs = removeStopWords(sadDocs);
sadDocs = removeShortWords(sadDocs,1);
sadBag = bagOfWords(sadDocs);
sadWords = topkwords(sadBag,10);
sadWordsImportance = zeros(1,10);
indexes = zeros(1,10);
for i = 1:10
    indexes(i) = find(strcmp([bag.Vocabulary], sadWords.Word(i)));
end
for i = 1:10
    if ~isempty(indexes)
        sadWordsImportance(i) = featureImportance(indexes(i));
    else
        % The case where the word is not in the vocabulary
        sadWordsImportance(i) = NaN;  
    end
end
%% Create a figure showing the importance and count of each sad word
figure;
nil = zeros(10, 1);
% Left y-axis for count
yyaxis left;
countBar = bar(categorical(sadWords.Word), [sadWords.Count nil], 1.1, 'grouped');
ylabel('Count');
% Right y-axis for importance
yyaxis right;
importanceBar = bar(categorical(sadWords.Word), [nil sadWordsImportance.'], 1.1, 'grouped');
ylabel('Importance');

title('Top Words in "Sad" Texts');
xlabel('Word');

%% Joy Vocab
joy = data.text(data.label == 1);
joyDocs = tokenizedDocument(joy);
joyDocs = removeStopWords(joyDocs);
joyDocs = removeShortWords(joyDocs,1);
joyBag = bagOfWords(joyDocs);
joyWords = topkwords(joyBag,10);
joyWordsImportance = zeros(1,10);
indexes = zeros(1,10);
for i = 1:10
    indexes(i) = find(strcmp([bag.Vocabulary], joyWords.Word(i)));
end
for i = 1:10
    if ~isempty(indexes)
        joyWordsImportance(i) = featureImportance(indexes(i));
    else
        % The case where the word is not in the vocabulary
        joyWordsImportance(i) = NaN;  
    end
end
%% Create a figure showing the importance and count of each joy word
figure;
nil = zeros(10, 1);
% Left y-axis for count
yyaxis left;
countBar = bar(categorical(joyWords.Word), [joyWords.Count nil], 1.1, 'grouped');
ylabel('Count');
% Right y-axis for importance
yyaxis right;
importanceBar = bar(categorical(joyWords.Word), [nil joyWordsImportance.'], 1.1, 'grouped');
ylabel('Importance');

title('Top Words in "Joy" Texts');
xlabel('Word');


%% Love vocab
love = data.text(data.label == 2);
loveDocs = tokenizedDocument(love);
loveDocs = removeStopWords(loveDocs);
loveDocs = removeShortWords(loveDocs,1);
loveBag = bagOfWords(loveDocs);
loveWords = topkwords(loveBag,10);
loveWordsImportance = zeros(1,10);
indexes = zeros(1,10);
for i = 1:10
    indexes(i) = find(strcmp([bag.Vocabulary], loveWords.Word(i)));
end
for i = 1:10
    if ~isempty(indexes)
        loveWordsImportance(i) = featureImportance(indexes(i));
    else
        % Handle the case where the word is not in the vocabulary
        loveWordsImportance(i) = NaN; 
    end
end
%% Create a figure showing the importance and count of each joy word
figure;
nil = zeros(10, 1);
% Left y-axis for count
yyaxis left;
countBar = bar(categorical(loveWords.Word), [loveWords.Count nil], 1.1, 'grouped');
ylabel('Count');
% Right y-axis for importance
yyaxis right;
importanceBar = bar(categorical(loveWords.Word), [nil loveWordsImportance.'], 1.1, 'grouped');
ylabel('Importance');

title('Top Words in "Love" Texts');
xlabel('Word');


%% Anger vocab
anger = data.text(data.label == 3);
angerDocs = tokenizedDocument(anger);
angerDocs = removeStopWords(angerDocs);
angerDocs = removeShortWords(angerDocs,1);
angerBag = bagOfWords(angerDocs);
angerWords = topkwords(angerBag,10);
angerWordsImportance = zeros(1,10);
indexes = zeros(1,10);
for i = 1:10
    indexes(i) = find(strcmp([bag.Vocabulary], angerWords.Word(i)));
end
for i = 1:10
    if ~isempty(indexes)
        angerWordsImportance(i) = featureImportance(indexes(i));
    else
        % Handle the case where the word is not in the vocabulary
        angerWordsImportance(i) = NaN;  
    end
end
%% Create a figure showing the importance and count of each anger word
figure;
nil = zeros(10, 1);
% Left y-axis for count
yyaxis left;
countBar = bar(categorical(angerWords.Word), [angerWords.Count nil], 1.1, 'grouped');
ylabel('Count');
% Right y-axis for importance
yyaxis right;
importanceBar = bar(categorical(angerWords.Word), [nil angerWordsImportance.'], 1.1, 'grouped');
ylabel('Importance');

title('Top Words in "Anger" Texts');
xlabel('Word');

%% Fear vocab
fear = data.text(data.label == 4);
fearDocs = tokenizedDocument(fear);
fearDocs = removeStopWords(fearDocs);
fearDocs = removeShortWords(fearDocs,1);
fearBag = bagOfWords(fearDocs);
fearWords = topkwords(fearBag,10);
fearWordsImportance = zeros(1,10);
indexes = zeros(1,10);
for i = 1:10
    indexes(i) = find(strcmp([bag.Vocabulary], fearWords.Word(i)));
end
for i = 1:10
    if ~isempty(indexes)
        fearWordsImportance(i) = featureImportance(indexes(i));
    else
        % Handle the case where the word is not in the vocabulary
        fearWordsImportance(i) = NaN;  
    end
end
%% Create a figure showing the importance and count of each joy word
figure;
nil = zeros(10, 1);
% Left y-axis for count
yyaxis left;
countBar = bar(categorical(fearWords.Word), [fearWords.Count nil], 1.1, 'grouped');
ylabel('Count');
% Right y-axis for importance
yyaxis right;
importanceBar = bar(categorical(fearWords.Word), [nil fearWordsImportance.'], 1.1, 'grouped');
ylabel('Importance');

title('Top Words in "Fear" Texts');
xlabel('Word');

%% Surprise vocab
surprise = data.text(data.label == 5);
surpriseDocs = tokenizedDocument(surprise);
surpriseDocs = removeStopWords(surpriseDocs);
surpriseDocs = removeShortWords(surpriseDocs,1);
surpriseBag = bagOfWords(surpriseDocs);
surpriseWords = topkwords(surpriseBag,10);
surpriseWordsImportance = zeros(1,10);
indexes = zeros(1,10);
for i = 1:10
    indexes(i) = find(strcmp([bag.Vocabulary], surpriseWords.Word(i)));
end
for i = 1:10
    if ~isempty(indexes)
        surpriseWordsImportance(i) = featureImportance(indexes(i));
    else
        % Handle the case where the word is not in the vocabulary
        surpriseWordsImportance(i) = NaN;  
    end
end
%% Create a figure showing the importance and count of each surprise word
figure;
nil = zeros(10, 1);
% Left y-axis for count
yyaxis left;
countBar = bar(categorical(surpriseWords.Word), [surpriseWords.Count nil], 1.1, 'grouped');
ylabel('Count');
% Right y-axis for importance
yyaxis right;
importanceBar = bar(categorical(surpriseWords.Word), [nil surpriseWordsImportance.'], 1.1, 'grouped');
ylabel('Importance');

title('Top Words in "Surprise" Texts');
xlabel('Word');






