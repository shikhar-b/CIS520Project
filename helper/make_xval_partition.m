function [part] = make_xval_partition(n, n_folds)
% MAKE_XVAL_PARTITION - Randomly generate cross validation partition.
%
% Usage:
%
%  PART = MAKE_XVAL_PARTITION(N, N_FOLDS)
%
% Randomly generates a partitioning for N datapoints into N_FOLDS equally
% sized folds (or as close to equal as possible). PART is a 1 X N vector,
% where PART(i) is a number in (1...N_FOLDS) indicating the fold assignment
% of the i'th data point.

    assert(n >= n_folds);
    batch_size = floor(n/n_folds);
    left_over = mod(n,n_folds);
    
    part = zeros(1,n);
    start = 1;
    if left_over>0
        part(1:left_over) = randperm(n_folds, left_over);
        start = left_over+1;
    end
    
    for i = 1:batch_size
        part(start : start+n_folds-1) = randperm(n_folds, n_folds);
        start = start+n_folds;
    end    
end