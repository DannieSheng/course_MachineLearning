% Binary Classifier based on Revelance Vector Machine (RVM)
% The Kernel function used in this code is SBL_KERNELFUNCTION from Microsoft Corporation.
% The SparseBayes calculation is based on SPARSEBAYES Matlab Toolbox.
%
% [EstParameters, Parameters] = TrainRVM(XEstimate, ClassLabels, Parameters)
% Input:
%       XEstimate: N x D training data
%       ClassLabels: N x 1 vector with labels for each training data
%       Parameters: Structure variable with Kernel type(str); BasisWidth(double); Iterations(int)
%
% Output:
%       EstParameters: structure variable. EstParameters.Parameters saved estimated relevant basis and
%       weights. EstParameters.HyperParameters saved the alpha and beta for
%       the model.
%       Parameters: The same as input parameters. Saved for the test.
%
% Author: Weihuang Xu
% University of Florida, Electrical and Computer Engineering

function [EstParameters, Parameters] = TrainRVM(XEstimate, ClassLabels, Parameters)

Kernel = Parameters.Kernel;
BasisWidth = Parameters.BasisWidth^(1/size(XEstimate, 2));
Iterations = Parameters.Iterations;

Basis = sbl_kernelFunction(XEstimate, XEstimate, Kernel, BasisWidth);

Likelihood = 'Bernoulli';

OPTIONS		= SB2_UserOptions('iterations',Iterations,...
							  'diagnosticLevel', 2,...
							  'monitor', 10);

SETTINGS	= SB2_ParameterSettings('NoiseStd',0.1);

[PARAMETER, HYPERPARAMETER, ] = ...
    SparseBayes(Likelihood, Basis, ClassLabels, OPTIONS, SETTINGS);

EstParameters.Parameters = PARAMETER;
EstParameters.HyperParameters = HYPERPARAMETER;
Parameters.BasisWidth = BasisWidth;
EstParameters.Parameters.RelevantX = XEstimate(PARAMETER.Relevant,:);

