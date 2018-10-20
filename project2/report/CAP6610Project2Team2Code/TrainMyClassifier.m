function [ YValidate, EstParameters, UpdatedParameters ] = TrainMyClassifier( XEstimate, XValidate, labelEstimate, labelValidate, Parameters, algorithm )
% Function used to train the classifier, for the algorithm, for the
% algorithms SVM and RVM, the training is completed by calling the function
% "binClassification" several times: i.e. the training procedure is
% completed by train a binary classifier every time; for the algorithm GPR,
% the completed training procedure is finished in the function "TrainMyGPR".
% 
%   Input: 
%          XEstimate:         estimate data used to train the classifier
%          XValidate:         validate data used to validate the classifier
%          labelEstimate:     label of the estimate data
%          labelValidate:     label of the validate data
%          Parameters:        parameters used when train the model
%          algorithm:         name of the algorithm used to train the model: RVM, SVM, GPR
%   Output:
%          YValidate:         the class labels for each sample in XValidate
%          EstParameters:     the estimated parameters trained from the estimate data
%          UpdatedParameters: the updated parameters (if any)
% Author: Hudanyun Sheng
% University of Florida, Electrical and Computer Engineering


trainGroup = Parameters.tG;
if strcmpi(algorithm, 'GPR')   
    [YValidate, EstParameters] = TrainMyGPR(XEstimate, XValidate, Parameters, labelEstimate);
else
    
	for idxGroup = 1:size(trainGroup,1) % binary classifiers
        xE{1,idxGroup}      = XEstimate(find(labelEstimate == trainGroup(idxGroup,1)),:);
        xE{1,idxGroup}      = [xE{1,idxGroup}; XEstimate(find(labelEstimate == trainGroup(idxGroup,2)),:)];
        
        Elabel{1, idxGroup} = labelEstimate(find(labelEstimate == trainGroup(idxGroup,1)),:);
        Elabel{1, idxGroup} = [Elabel{1, idxGroup}; labelEstimate(find(labelEstimate == trainGroup(idxGroup,2)),:);];
        
        xV{1,idxGroup}      = XValidate(find(labelValidate == trainGroup(idxGroup,1)),:);
        xV{1,idxGroup}      = [xV{1,idxGroup}; XValidate(find(labelValidate == trainGroup(idxGroup,2)),:)];
        
        Vlabel{1, idxGroup} = labelValidate(labelValidate == trainGroup(idxGroup,1),:);
        Vlabel{1, idxGroup} = [Vlabel{1, idxGroup}; labelValidate(labelValidate == trainGroup(idxGroup,2),:)];
        [ YValidate{1, idxGroup}, EstParameters{1, idxGroup}, UpdatedParameters{1, idxGroup}] = binClassification(xE{1,idxGroup}, xV{1,idxGroup}, Elabel{1, idxGroup}, Vlabel{1, idxGroup}, Parameters, algorithm); 
	end

end
UpdatedParameters{1, size(trainGroup,1)+1}= trainGroup;
end

function [YValidate, EstParameters, updatedParameters] = binClassification(xE, xV, yE, yV, Parameters, algorithm)
%  function used to train a binary classifier, the procudure is completed by
%  calling function either "TrainRVM" or "TrainSVM", based on the input
%  value "algorithm"
% 
% Input:  
%          xE:         estimate data used to train the classifier
%          xV:         validate data used to validate the classifier
%          yE:     label of the estimate data
%          yV:     label of the validate data
%          Parameters:        parameters used when train the model
%          algorithm:         name of the algorithm used to train the model: RVM, SVM, GPR
% Output:
%          YValidate:         the class labels for each sample in XValidate
%          EstParameters:     the estimated parameters trained from the estimate data
%          UpdatedParameters: the updated parameters (if any)
minOriLb = min(yE);
maxOriLb = max(yE);
ClassLabels(yE == minOriLb,:) = 0;
ClassLabels(yE == maxOriLb,:) = 1;

if strcmpi(algorithm, 'RVM')
    [EstParameters, updatedParameters] = TrainRVM( xE, ClassLabels, Parameters);
    YValidate = TestRVM(xV, updatedParameters, EstParameters);
elseif strcmpi(algorithm, 'SVM')
    EstParameters  = TrainSVM(xE, ClassLabels, Parameters);   
    [~, YValidate] = TestSVM(xV, EstParameters);
    updatedParameters = [];    
end
YValidate(YValidate == 1,:) = max(yV);
YValidate(YValidate == 0,:) = min(yV);

end
