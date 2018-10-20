function [YTest] = TestMyGPR(XTest, Parameters, EstParameters)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% INPUT:XTest(N*60),Parameters(3*1,sigma and l,threshold),EstParameters:struct
%%% OUPUT:YTest(N*(Nc+1))
%%% Example call:[YTest] = TrainMyGPC(XTest, Parameters, ClassLabels);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Initialization
gprMdl         = EstParameters.classifers;
NumClass       = EstParameters.NumClass;
C              = EstParameters.C;
uniClassLabels = EstParameters.uniClassLabels;
% distances = EstParameters.distances;
YTest          = zeros(size(XTest,1),NumClass+1);
means          = EstParameters.means;

%% Classification by voting
for n = 1:size(C,1)
    column1 = find(uniClassLabels == C(n,1));
    column2 = find(uniClassLabels == C(n,2));
    tempResult = 1./(1+exp(-predict(gprMdl{n,1},XTest)));
    YTest( find(tempResult>=0.5),column1 ) =  YTest( find(tempResult>=0.5),column1 ) + 1;
    YTest( find(tempResult<0.5), column2 ) =  YTest( find(tempResult<0.5), column2 ) + 1;
end

% if isempty(Parameters)
%     threshold = 1.5;
% else
%     threshold = Parameters(3,1);
% end

%% Determing whether the test data doesn't belong to any class
% for n = 1:size(XTest,1)
%     tempResult = find(YTest(n,:) == max(YTest(n,:)));
%     tempNorm = norm(XTest(n,:) - means(tempResult,:));
%     if (size(tempResult,2)~=1) || (tempNorm> threshold*distances(tempResult)) %%% could not classify it
%         YTest(n,1:NumClass) = -1;
%         YTest(n,NumClass+1) = 1;
%     else    
%         YTest(n,:) = -1;
%         YTest(n,tempResult) = 1;
%     end
% end
