function [YValidate, EstParameters] = TrainMyGPR(XEstimate, XValidate, Parameters, ClassLabels)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Input:XEstimate(N*60),XValidate(N*60),Parameters(3*1,sigma and l,threshold),
%%% ClassLabels(N*5)
%%% Output:YValidate(N*6),EstParameters(struct)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Example call:[YValidate, EstParameters] = TrainMyGPC(XEstimate, XValidate, Parameters, ClassLabels);
%%% Author: Yuanhang Lin
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Initialization
% [r,ClassLabels,v] = find(ClassLabels==1);
NumClass        = size(unique(ClassLabels),1);
uniClassLabels  = unique(ClassLabels);
C               = nchoosek(unique(ClassLabels),2);
gprMdl          = cell(size(C,1),1);
classifier_info = cell(size(C,1),1);

%% Trainning
for n = 1:size(C,1)
    classifier_info{n,1} = ['class ',num2str(C(n,1)),' vs class ',num2str(C(n,2))];
    tempDataC1           = XEstimate(find(ClassLabels==C(n,1)),:);
    tempDataC2           = XEstimate(find(ClassLabels==C(n,2)),:);
    tempData             = [tempDataC1;tempDataC2];
    tempLabel            = [ones(size(tempDataC1,1),1);-ones(size(tempDataC2,1),1)];
    clear tempDataC1 tempDataC2
%     if isempty(Parameters)
        PHI              = [mean(std(tempData));std(tempLabel)/sqrt(2)];
%     else
%         PHI              = Parameters(1:2,:);
%     end
%     gprMdl{n,1} = compact(fitrgp(tempData,tempLabel,'KernelParameters',PHI));
    gprMdl{n,1} = compact(fitrgp(tempData,tempLabel,'KernelParameters',PHI, 'KernelFunction',Parameters.Kernel));
end

%% Calculating average distances
distances = [];
means = [];
for n = 1:size(uniClassLabels,1)
    tempData     = XEstimate(ClassLabels==n,:);
    tempMean     = mean(tempData);
    tempDistance = tempData-tempMean;
    tempNorm     = [];
    for k = 1:size(tempDistance,1)
        tempDistance(k,:);
        tempNorm = [tempNorm;norm(tempDistance(k,:))];
    end
    distances    = [distances;mean(tempNorm)];
    means        = [means;tempMean];
end

%% 
EstParameters.classifers      = gprMdl;
EstParameters.NumClass        = NumClass;
EstParameters.uniClassLabels  = uniClassLabels;
EstParameters.classifier_info = classifier_info;
EstParameters.C               = C;
EstParameters.distances       = distances;
EstParameters.means           = means;
YValidate                     = TestMyGPR(XValidate, Parameters, EstParameters);

end