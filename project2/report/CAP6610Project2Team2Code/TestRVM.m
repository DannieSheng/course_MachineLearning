% Binary Classifier based on Revelance Vector Machine (RVM)
% The Kernel function used in this code is SBL_KERNELFUNCTION from Microsoft Corporation.
% The SparseBayes calculation is based on SPARSEBAYES Matlab Toolbox.
%
% [YTest] = TestRVM(XTest, Parameters, EstParameters)
% Input:
%       XTest: N x D test data
%       Parameters: Structure variable with Kernel type(str); BasisWidth(double); Iterations(int)
%       EstParameters: structure variable. Estimated parameters from training. 
%
% Output:
%      Ytest: N x 1 vector. The estimated labels fot each test data.
%
% Author: Weihuang Xu
% University of Florida, Electrical and Computer Engineering

function [YTest] = TestRVM(XTest, Parameters, EstParameters)

Kernel = Parameters.Kernel;
BasisWidth = Parameters.BasisWidth;

Basis = sbl_kernelFunction(XTest, EstParameters.Parameters.RelevantX, Kernel, BasisWidth);

output = Basis * EstParameters.Parameters.Value;
YTest = double(SB2_Sigmoid(output)>0.5);