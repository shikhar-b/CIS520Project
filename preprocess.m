function data=preprocess(inputs)
    X_tweets = inputs(:,22:2021);
    X_demo = inputs(:,1:21);
    [coeff, score, latent, tsquared, explained] = pca(X_tweets);
    X_tweets_reduced = score(:,1:25);
    data = [X_demo, X_tweets_reduced];
    