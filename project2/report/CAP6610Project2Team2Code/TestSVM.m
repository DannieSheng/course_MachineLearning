function [YTest, ClassLabelsTest] = TestSVM(XTest, Mdl)
% SVM classifier training function for Project 2 in CAP 6610 
% 
% Syntax: [YTest] = TestSVM(XTest, Parameters, EstParameters)
% 
% Inputs:
%     XTest - Test set 
%     Mdl - Trained model
%     
% Outputs:
%     YTest - Vector representing probability [0,1] of classes
%     ClassLabelsTest - Class labels (1-6) for each sample in test set (6
%     corresponds to the unknown class)
%
% Author: Pallavi Raiturkar
% University of Florida, Computer and Information Science and Engineering

[ClassLabelsTest,~,~,YTest] = predict(Mdl,XTest);

%include unknown class

% ThresholdUnknownClass = 0.5;
% for i = 1:size(XTest,1)
%     if(max(YTest(i,:))<ThresholdUnknownClass)
%         ClassLabelsTest(i) = 6;
%     end
% end

end