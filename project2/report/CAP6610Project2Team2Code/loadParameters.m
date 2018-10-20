function Parameters = loadParameters(  )
%function to load(change) parameters based on the algorithm used
%   Input:  name of algorithm: RVM, SVM or GPR
%   Output: Parameters to be used later on, a structure

Parameters.dataName       = 'Proj2FeatVecsSet1.mat';
Parameters.labelName      = 'Proj2TargetOutputsSet1.mat';
Parameters.numFold        = 5;
Parameters.RVM.Kernel     = 'gauss';
Parameters.RVM.BasisWidth = 0.5;
Parameters.RVM.Iterations = 100;
Parameters.SVM.Kernel     = 'gauss';
Parameters.SVM.Iterations = 100;
Parameters.GPR.Kernel     = 'SquaredExponential';
end

