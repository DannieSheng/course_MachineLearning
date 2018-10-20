function [ EstParameters, YValidate, confM, avgAcc, ConfMatrix, avgAccAll, idxF ] = MyCrossValidate( XTrain, ClassLabels, Nf, algorithm, Parameters )
%Cross validation function used for the course project CAP6610
%   Detailed explanation goes here
% Input:  
%         XTrain:        N*D data matrix
%         ClassLabels:   N*1 labels corresponding to the XTrain
%         Nf:            number of folds
%         algorithm:     algorithm used to train the classifiers
%         Parameters:    parameters used for to train/test the classifiers
% Output: 
%         EstParameters: a cell array, where every cell represents the estimated parameters for the corresponding fold
%         YValidate:     a cell array, where every cell represents the class labels for each sample of the validation data of the corresponding fold
%         confM:         a cell array, where every cell represents the confusion matrix for the corresponding fold
%         avgAcc:        a cell array, where every cell represents the classification accuracy for the corresponding fold
%         ConfMatrix:    an overall confusion matrix for all of XTrain
%         avgAccAll:     the classification accuracy for all of XTrain
%         idxF:          the index of the best fold estimated
% Author: Hudanyun Sheng
% University of Florida, Electrical and Computer Engineering

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


ListLabel = unique(ClassLabels);
nClass = length(unique(ListLabel));

%%%%%%%%%%%%%%%%%%% Set the cross validation groups %%%%%%%%%%%%%%%%%%%%
for i = 1:nClass
    tempIndex = [];
    N(i) = length(find(ClassLabels == ListLabel(i)));
    for j = 1:(N(i)/Nf)
        tempIndex = [tempIndex; randperm(Nf)'];
    end
    dataIndex(find(ClassLabels == ListLabel(i)),:) = tempIndex;
end

temp = 1;
for idxC1 = 1:nClass
    for idxC2 = idxC1+1:nClass
        trainGroup(temp,1) = idxC1;
        trainGroup(temp,2) = idxC2;
        temp = temp+1;
    end
end
allLabel = [];
allLabelV = [];

for iSet = 1:Nf  % cross validation
	V{1, iSet} = XTrain(dataIndex == iSet,:);
    LabelV{1, iSet} = ClassLabels(dataIndex == iSet);
    allLabelV = [allLabelV; LabelV{1, iSet}];
    E{1, iSet} = XTrain(dataIndex ~= iSet,:);
    LabelE{1, iSet} = ClassLabels(dataIndex ~= iSet);
    distances = zeros(1,nClass);
    for iC = 1:nClass
        daTemp = E{1, iSet}(find(LabelE{1, iSet} == iC),:);
        mu{1, iSet}(iC,:) = mean(daTemp);
        sigma{1, iSet}{1,iC} = cov(daTemp);
        for n = 1:size(daTemp, 1)
            distances(1, iC) = distances(1, iC) + Mdistance(daTemp(n,:),mu{1, iSet}(iC,:),sigma{1, iSet}{1,iC});
        end
        distances(1, iC) = distances(1, iC)/size(daTemp, 1);
    end
    Parameters.tG = trainGroup;
    
   [ ~, EstParameters{1, iSet}, updatedParameters{1, iSet} ] = TrainMyClassifier( E{1, iSet}, V{1, iSet}, LabelE{1, iSet},LabelV{1, iSet}, Parameters, algorithm );
    XVali = V{1, iSet};
    yVali = TestMyClassifier( XVali, updatedParameters{1, iSet}, EstParameters{1, iSet}, nClass, algorithm );
    
    YValidate{1, iSet} = determineLabel(V{1, iSet}, yVali, mu{1, iSet}, sigma{1, iSet}, distances, nClass);
    for n = 1:size(YValidate{1, iSet}, 1)
        t = find(YValidate{1, iSet}(n,:) == 1);
        yy(n,:) = t;
    end
    allLabel = [allLabel;  yy];
    [ confM{1, iSet}, avgAcc(iSet) ] = MyConfusionMatrix( LabelV{1, iSet},yy );
    fprintf(['\nThe confusion matrix for fold ' num2str(iSet) ' is \n'])
    disp(confM{1, iSet})
	fprintf(['\nThe average accuracy for fold ' num2str(iSet) ' is ' num2str(avgAcc(iSet))])
%     aaaLV = zeros(6,5000 );
%     aaaL = zeros(6,5000 );
%     for i = 1:5000
%         aaaLV(LabelV{1, iSet}(i,:),i) = 1;
%         aaaL(yy(i,:),i) = 1;
%     end
%     plotconfusion(aaaLV, aaaL)
%     saveas(gcf, ['confu_' algorithm '_fold_' num2str(iSet) '.jpg'])
end
idxF = find(avgAcc == max(avgAcc));
YVali = TestMyClassifier( cell2mat(V'), updatedParameters{1, idxF}, EstParameters{1, idxF}, nClass, algorithm);
YYTest = determineLabel(cell2mat(V'), YVali, mu{1, iSet}, sigma{1, iSet}, distances, nClass);
for n = 1:size(YYTest, 1)
	t = find(YYTest(n,:) == 1);
	YY(n,:) = t;
end
[ ConfMatrix, avgAccAll ] = MyConfusionMatrix( allLabelV, YY );
% aaLV = zeros(6, length(allLabelV));
% aaL = aaLV;
% for i =1:length(allLabelV)
% 	aaLV(allLabelV(i,:),i) = 1;
% 	aaL(YY(i,:),i) = 1;
% end
fprintf('\nThe overall confusion matrix is \n')
disp(ConfMatrix)
fprintf(['\nThe overall accuracy is ' num2str(avgAccAll)])
% plotconfusion(aaLV, aaL)
% saveas(gcf, ['confu_' algorithm '_all.jpg'])
end


function finalLabel = determineLabel(xtest, label, mu, sigma, distances, nClass)

% the function used to determine the label for the test data based on
% voting as well as the distances to the center of the corresponding train
% class, taking the case of the data belong to non-class into consideration

for n = 1:size(xtest, 1)
	tempResult = find(label(n,:) == max(label(n,:)));
	for i = 1:length(tempResult)
        tempNorm(i) = Mdistance(xtest(n,:),mu(tempResult(i),:),sigma{1, tempResult(i)}); % Mahalanobis distance
%     tempNorm = norm(XVali(n,:) - mu{1, iSet}); % L2 norm
            if (size(tempResult,2)~=1) || (tempNorm(i)> 1.5*distances(1,tempResult(i))) %%% could not classify it
                finalLabel(n,1:nClass) = -1;
                finalLabel(n,nClass+1) = 1;
            else    
                finalLabel(n,:) = -1;
                finalLabel(n,tempResult) = 1;
            end
	end
end

end
