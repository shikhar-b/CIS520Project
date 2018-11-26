function X = standardize(x_old)
    mean_X = mean(x_old); %// Find mean of each column
    std_X = std(x_old); %// Find std. dev. of each column
    X = bsxfun(@minus, x_old, mean_X); %// Subtract each column by its respective mean
    X = bsxfun(@rdivide, X, std_X); %// Take each column and divide by its respective std dev.
