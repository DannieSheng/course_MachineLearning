function [Rx] = Mdistance(input,mu,cov_train)


Rx = sqrt((input-mu)* inv(cov_train)*(input-mu)');