function [ confu_M, avgAccuracy ] = MyConfusionMatrix( TrueLabel, PredictedLabel )
%Function to compute confusion matrix
%   Input: 
%       predicted label: label predicted by the discrimint functions
%       true label:      the ground truth from the data set
%   Output:
%       confu_M:         the confusion matrix calculated based on the TrueLabel and PredictedLabel
%       avgAccuracy:     the overall classification accuracy
% Author: Hudanyun Sheng
% University of Florida, Electrical and Computer Engineering

listClass = unique(TrueLabel);
confu_M   = zeros(length(listClass)+1, length(listClass)+1);
for i = 1:length(listClass)
    idx  = find(TrueLabel == i);
    temp = PredictedLabel(idx);
    listTemp = unique(temp);
    for j = 1:length(listTemp)
        confu_M(i, listTemp(j)) = confu_M(i, listTemp(j))+length(find(temp == listTemp(j)));
    end
end
avgAccuracy = length(find(TrueLabel == PredictedLabel))/length(TrueLabel);
end

