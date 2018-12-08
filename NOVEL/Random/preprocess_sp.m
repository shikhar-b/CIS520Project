function [train_data, test_data] =preprocess_sp(inputs, test_inputs, n_comp)
    inputs = zscore(inputs);
    test_inputs = zscore(test_inputs);
    X_tweets = inputs(:,16:2015);
    X_demo = inputs(:,1:15);
    [coeff, score, latepnt, tsquared, explained] = pca(X_tweets);
    n_comp = min(n_comp, size(coeff,2));
    coeff_selected = coeff(:, 1:n_comp);
    X_tweets_reduced = X_tweets * coeff_selected;
    train_data = [X_demo, X_tweets_reduced];
    test_data = [test_inputs(:,1:15), test_inputs(:,16:2015)*coeff_selected];

