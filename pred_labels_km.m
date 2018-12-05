function pred_labels=pred_labels_km(train_inputs,train_labels,test_inputs)

addpath('Analysis');
addpath('Discriminative');
addpath('Generative');
addpath('InstanceBased');
addpath('helper');
addpath('GLM');
addpath('glmnet_matlab');
addpath('LARS');
addpath('lars_lasso/lars');

[X_tp, X_tep] = preprocess(train_inputs, test_inputs, 150);
[idxs, kc] = KM(X_tp);
distances = zeros(size(X_tep,1),size(kc,1));
predictions = zeros(size(X_tep,1),size(train_labels,2),size(kc,1));

for i = 1:size(kc,1)
    clust = idxs==i;
 	clust_train = train_inputs(clust,:);
 	clust_train_labels = train_labels(clust,:);
    predictions(:,:,i) = predict_labels(clust_train, clust_train_labels, test_inputs);
    distances(:,i) = sum(sqrt((X_tep - kc(i)).^2), 2);
end

sum_dist = sum(distances, 2);

pred_labels = zeros(size(test_inputs,1),size(train_labels,2));

for i = 1:size(test_inputs,1)
    for j = 1:size(kc,1)
        pred_labels(i,:) = pred_labels(i,:) + predictions(i,:,j).*distances(i,j);
    end
    pred_labels(i,:) = pred_labels(i,:)./sum_dist(i);
end

end

