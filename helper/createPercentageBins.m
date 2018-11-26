function X = createPercentageBins(x_old)
    cols  = [1,2,3,4,5,6,7,10,11,12,20];
    X = x_old;
    for i = cols
        X(:, i) = round(x_old(:, i), 0);
    end
    