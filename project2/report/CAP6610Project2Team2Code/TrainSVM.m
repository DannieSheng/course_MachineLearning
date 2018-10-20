function [Mdl] = TrainSVM(XEstimate, ClassLabelsEstimate, Parameters)

% SVM classifier training function for Project 2 in CAP 6610 
% 
% Syntax: [EstParameters, Parameters] = TrainSVM(XEstimate, ClassLabels, Parameters)
% 
% Inputs:
%     XEstimate - Estimation set 
%     Parameters - Cell array = {KernelType, IterLimit}
%     ClassLabelsEstimate - Class labels for estimation set
%     
% Outputs:
%     Mdl - trained SVM model
% 
% Author: Pallavi Raiturkar
% University of Florida, Computer and Information Science and Engineering

switch nargin
    case 2
%         KernelType = 'gaussian';
        KernelType = 'gauss';
        IterLimit = 15000;
    case 3
        switch size(Parameters,2)
            case 1
%                 KernelType = char(Parameters(1,:));
                KernelType = Parameters.Kernel;
                IterLimit = 15000;
            case 2
%                 KernelType = char(Parameters(1,:));
                KernelType = Parameters.Kernel;
%                 IterLimit = Parameters(2,:);
                IterLimit = Parameters.Iterations;
        end
end


t = templateSVM('KernelFunction',KernelType,'IterationLimit',IterLimit);
Mdl = fitcecoc(XEstimate,ClassLabelsEstimate,'Coding','allpairs','Learners',t,'FitPosterior',1);



end