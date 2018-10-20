script_project document:
The parameters can be changed in the function called "loadParameters"
By calling the function "MyCrossValidate", the data is Randomly partitioned into Nf
pairs of Estimation and Validation Sets, the parameters and hyper-parameters are estimated
using estimate sets, and the confusion matrices and accuracy are evaluated on the validation
sets. The cross-validation procedure is completed by calling function "TrainMyClassifier", 
which includes three separate function based on algorithms SVM, RVM, GPR, and also the
validate labels are gained by calling function "TestMyClassifier", which includes three
separate function based on algorithms SVM, RVM, GPR.



RVM Binary Classifier document:

There are two main functions: TrainRVM.m and TestRVM.m for training and testing, respectively.

[EstParameters, Parameters] = TrainRVM(XEstimate, ClassLabels, Parameters)
% Input:
%       XEstimate: N x D training data
%       ClassLabels: N x 1 vector with labels for each training data
%       Parameters: Structure variable with Kernel type(str); BasisWidth(double); Iterations(int)
%
% Output:
%       EstParameters: Structure variable. EstParameters.Parameters saved estimated relevant basis and
%       		weights. EstParameters.HyperParameters saved the alpha and beta for the model.
%       Parameters: The same as input parameters. Saved for the test.

Note: 1. The input variable 'Parameter' is generated from the funcion 'loadParameters.m'. 
      2. The default kernel function is Gaussian. You can change in the funcion 'loadParameters.m' when you generate the Parameters.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[YTest] = TestRVM(XTest, Parameters, EstParameters)
% Input:
%       XTest: N x D test data
%       Parameters: Structure variable with Kernel type(str); BasisWidth(double); Iterations(int)
%       EstParameters: structure variable. Estimated parameters from training. 
%
% Output:
%      Ytest: N x 1 vector. The estimated labels fot each test data.

Note: 1. The second input 'Parameters' is the same as the 'Parameters' in function TrainRVM.m. It is generated from  the funcion 
         'loadParameters.m'. Those are the constant parameters instead of learned hyper-parameters.
      2. The third input 'EstParameters' contains the learned hyper-parameters for the model including relevant vectors and their values.  



GPR Classifier document: 
[For GPR there are 3 functions: GPRParameters, TestMyGPC and TrainMyGPC.

(1) GPRParameters
Example call: [Parameters] = GPRParameters(XEstimate, ClassLabels)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  This function sets the parameters to be used for GPR classification.
%  Input:XEstimate(N*60), ClassLabels(N*1)
%  Output:
%  Parameters - struct - The struct contatins the following field
%                        1. sigma_f: signal standard deviation
%                        2. sigma_l: characteristic length scales
%  Note: All the parameters will be adjusted according to the Training set
%  By default, sigma_f = mean(std(XEstimate)),
%              sigma_l = std(ClassLabels)/sqrt(2)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

(2) TrainMyGPC
Example call: [YValidate, EstParameters] = TrainMyGPC(XEstimate, XValidate, Parameters, ClassLabels)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input: XEstimate(N*60), XValidate(N*60), Parameters(struct), ClassLabels(N*1)
% Output: YValidate(N*6), EstParameters(struct)
% In this function we apply one vs one method to train classifiers.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

(3) TestMyGPC 
Example call: [YTest] = TestMyGPC(XTest, Parameters, EstParameters)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input: XTest(N*60), Parameters(struct), EstParameters(struct)
% Output: YTest(N*(Nc+1))
% Since matlab build-in GPR algorithm ouputs y, not p(y), we apply sigmoid function
% to do classification. We apply 10 classifiers and voting rule to determine which
% class does the data point belong to.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



SVM Classifier document: 
[Mdl] = TrainSVM(XEstimate, ClassLabelsEstimate, Parameters)

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

[YTest] = TestMyGPR(XTest, Parameters, EstParameters)
% INPUT:XTest(N*60),Parameters(3*1,sigma and l,threshold),EstParameters:struct
% OUPUT:YTest(N*(Nc+1))
% Example call:[YTest] = TrainMyGPC(XTest, Parameters, ClassLabels);