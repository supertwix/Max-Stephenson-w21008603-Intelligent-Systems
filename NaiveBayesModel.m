clc; clear; % clear screen and clear workspace
fidPositive = fopen(fullfile('opinion-lexicon-English','positive-words.txt'));
C = textscan(fidPositive,'%s','CommentStyle',';'); %skip comment lines 
wordsPositive = string(C{1});
fclose all;

%puts the words into a hash table
words_hash = java.util.Hashtable;
[possize, ~] = size(wordsPositive); 
for ii = 1:possize
words_hash.put(wordsPositive(ii,1),1);
end

%same as above but for negative words
fidNegative = fopen(fullfile('opinion-lexicon-English','negative-words.txt'));
C = textscan(fidNegative,'%s','CommentStyle',';'); 
wordsNegative = string(C{1});
fclose all;

%adds them to the same list
[negsize, ~] = size(wordsNegative); 
for ii = 1:negsize
words_hash.put(wordsNegative(ii,1),-1);
end

%reads the file and getes the data from the csv
filename = "IMDBdatasetSMALL.csv";
%filename = "IMDBdatasetLARGE.csv";
dataReviews = readtable(filename,'TextType','string'); 
textData = dataReviews.review; %get data from the 'Review' column
actualScore = dataReviews.sentiment; %get data from the 'Sentiment' column

%PREPROCESSING THE DATA
% tokenize the text
sents = tokenizedDocument(textData);
% tonvert to lowercase
sents = lower(sents);
% erase punctuation
sents = erasePunctuation(sents);

fprintf('File: %s, Sentences: %d\n', filename, size(sents)); %prints the processed data

sentimentScore = zeros(size(sents));
for ii = 1 : sents.length
    docwords = sents(ii).Vocabulary;
    for jj = 1 : length(docwords)
        if words_hash.containsKey(docwords(jj))
            sentimentScore(ii) = sentimentScore(ii) + words_hash.get(docwords(jj));
        end
    end
    fprintf('Sent: %d, words: %s, FoundScore: %d, GoldScore: %d\n', ii, joinWords(sents(ii)), sentimentScore(ii), actualScore(ii));
end

sentimentScore(sentimentScore > 0) = 1;   %any score larger than 0 is positive
sentimentScore(sentimentScore < 0)= -1;   %counts any neutral or below negative score as Negative

notfound = sum(sentimentScore == 0);
covered = numel(sentimentScore)- notfound;
tp=0; tn=0; fn=0; fp=0; count=0;
%calculates the true positive and true negatives
for i=1:length(actualScore)
    if sentimentScore(i)==1 && actualScore(i)==1
        tp=tp+1; count=count+1; %true positive
    elseif sentimentScore(i)==0 && actualScore(i)==1
        fp=fp+1; %false positive
    elseif sentimentScore(i)==-1 && actualScore(i)==0
        tn=tn+1; count=count+1;%true negative
    else
        fn=fn+1;%false negative
    end
end
%calculates statistics
accuracy = (tp+tn)*100/covered 
%precision gives result between 0.0 and 1.0
precision = (tp/(tp+fp))
%recall gives result between 0.0 and 1.0
recall = (tp/(tp+fn))
coverage=covered*100/numel(sentimentScore)
%f1 score
f1score = 2*((precision*recall)/(precision+recall))
fprintf('TP: %d TN: %d FP: %d FN: %d Missed: %d', tp, tn, fp, fn, notfound)

load ('wordembedding.mat');
words = [wordsPositive;wordsNegative]; 
labels = categorical(nan(numel(words),1)); 
labels(1:numel(wordsPositive)) = "Positive";
labels(numel(wordsPositive)+1:end) = "Negative";

data = table(words,labels,'VariableNames',{'Word','Label'});
idx=~isVocabularyWord(emb,data.Word);
data(idx,:) = [];

numWords = size(data,1);
cvp = cvpartition(numWords,'HoldOut',0.01); %holdout fewer if applying model
dataTrain = data(training(cvp),:); 
dataTest = data(test(cvp),:);
%Convert the words in the training data to word vectors using word2vec. 
wordsTrain = dataTrain.Word;
XTrain = word2vec(emb,wordsTrain);
YTrain = dataTrain.Label;

%fitcnb trains the Naive Bayes classifier
model = fitcnb(XTrain,YTrain);
wordsTest = dataTest.Word;
XTest = word2vec(emb,wordsTest); 
YTest = dataTest.Label;
[YPred,scores] = predict(model,XTest);
figure
confusionchart(YTest,YPred, 'ColumnSummary','column-normalized'); 

