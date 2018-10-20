function YTest = TestMyClassifier( XTest, Parameters, EstParameters, nClass, algorithm)
% function to gain class labels for each sample in XTest, for algorithms SVM 
% as well as RVM, the test procedure is completed by test data going through 
% every two-class classifiers, which is realized by calling the function
% "binTest" several times; for the algorithm GPR, the test procedure is
% completed by one call of the function "TestMyGPR", since the procedure is
% completed inside the function
% 
%   Input: 
%            XTest:         data to be test on
%            Parameters:    the parameters (if any)
%            EstParameters: the model trained
%            nClass:        number of classes (which should be knwon based on training data)
%            algorithm:     the name of the algorithm chosen to test
%   Outtput: 
%            YTest:         the N*1 class labels for each sample in XTest

  
estParameters = EstParameters;
trainGroup    = Parameters{1,11};
YTest = zeros(size(XTest, 1), nClass);
if strcmpi(algorithm, 'GPR')  
    parameters    = [];
    YTest = TestMyGPR(XTest, parameters, estParameters);
else
    parameters = Parameters;
    for idxGroup = 1:size(estParameters,2)
        column1 = trainGroup(idxGroup,1);
        column2 = trainGroup(idxGroup,2);
        tempResult = binTest(XTest, parameters{1, idxGroup}, estParameters{1, idxGroup}, algorithm);
        YTest(find(tempResult == 0), column1) = YTest(find(tempResult == 0), column1)+1;
        YTest(find(tempResult == 1), column2) = YTest(find(tempResult == 1), column2)+1;
    end
end


end

function testResult = binTest(XTest, updatedParameters, EstParameters, algorithm)
% function to gain binary class labels for each same in XTest, when
% algorithm is either RMV or SVM
%   Input: 
%            XTest:         data to be test on
%            Parameters:    the parameters (if any)
%            EstParameters: the model trained
%            algorithm:     the name of the algorithm chosen to test
%   Outtput: 
%            testResult:    the binary class labels for each sample in XTest

if strcmpi(algorithm, 'RVM')
    testResult = TestRVM(XTest, updatedParameters, EstParameters);
else
    [~, testResult] = TestSVM(XTest, EstParameters);
end
end