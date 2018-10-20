dbstop if error
clear; close all; clc

algorithm = 'GPR';
Parameters = loadParameters( );

var_struct = load(Parameters.dataName);
name_cell  = fieldnames(var_struct);
XTrain     = getfield(var_struct, char(name_cell)); 

var_struct = load(Parameters.labelName);
name_cell  = fieldnames(var_struct);
labels     = getfield(var_struct, char(name_cell)); 

if size(labels,2)~=1
	for i = 1:size(labels, 1)
        ClassLabels(i,:) = find(labels(i,:)==1);
	end
else
    ClassLabels = labels;
end

Nf = Parameters.numFold;
tic
[ EstParameters, YValidate, confM, avgAcc, confuM, avgAccAll, idxF ] = MyCrossValidate( XTrain, ClassLabels, Nf, algorithm, Parameters.(char(algorithm)) );
toc
save(['results_' algorithm '.mat'], 'EstParameters', 'confM', 'avgAcc', 'confuM', 'avgAccAll', 'idxF' )