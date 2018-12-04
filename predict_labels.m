function pred_labels=predict_labels(train_inputs,train_labels,test_inputs)

addpath('Analysis');
addpath('Discriminative');
addpath('Generative');
addpath('InstanceBased');
addpath('helper');
addpath('GLM');
addpath('glmnet_matlab');
addpath('LARS');
addpath('lars_lasso/lars');
%[train_inputs_pre_r, test_inputs_pre_r] = preprocess(train_inputs, test_inputs, 150);

%[train_inputs_pre_g, test_inputs_pre_g] = preprocess(train_inputs, test_inputs, 110);

%[train_inputs_pre_ra, test_inputs_pre_ra] = preprocess(train_inputs, test_inputs, 75);

%input = [train_inputs;test_inputs];
%idxs = KM(input);
%pred_labels = zeros(size(test_inputs,1),9);
%for i=1:5
%	clust = idxs==i;
%	clust_train = train_inputs(clust(1:size(train_inputs,1)),:);
%	clust_train_labels = train_labels(clust(1:size(train_labels,1)),:);
%	clust_test = test_inputs(clust(size(train_inputs,1)+1:end),:);
%	disp(size(clust_test));
%	if i==1 | i==3 | i==5
%		pred_labels(clust(size(train_inputs,1)+1:end),:) = ridgeResidual(clust_train, clust_train_labels, clust_test);
%	elseif i==2
%		pred_labels(clust(size(train_inputs,1)+1:end),:) = glmGpr(clust_train, clust_train_labels,clust_test);
%	else
%		pred_labels(clust(size(train_inputs,1)+1:end),:) = glmLARS(clust_train, clust_train_labels, clust_test);
%	end
%end


pred_labels_1 = glmRidge(train_inputs, train_labels, test_inputs);
%pred_labels_2 = GPRResidual(train_inputs, train_labels, test_inputs);
pred_labels_2 = glmglmRidge(train_inputs, train_labels,test_inputs);
pred_labels_3 = GPRResidual(train_inputs, train_labels, test_inputs);
%load('pred_labels_1.mat');
%load('pred_labels_5.mat');
%save('pred_labels_1.mat','pred_labels_1');
%save('pred_labels_2.mat','pred_labels_2');
%save('pred_labels_3.mat','pred_labels_3');
%save('pred_labels_4.mat','pred_labels_4');
%save('pred_labels_5.mat','pred_labels_5');
%save('pred_labels_6.mat','pred_labels_6');
%pred_labels_4 = RandomForest(train_inputs_pre_ra, train_labels, test_inputs_pre_ra);
%pred_labels_3 = NNtest(train_inputs, train_labels, test_inputs);
%pred_labels_4 = GRNN(train_inputs, train_labels, test_inputs);
%pred_labels = (pred_labels_1+pred_labels_2+pred_labels_3+pred_labels_4+pred_labels_5)./5;
%pred_labels=randn(size(test_inputs,1),size(train_labels,2));
%pred_labels = pred_labels_6;
pred_labels = (0.3.*pred_labels_1 + 0.3.*pred_labels_2 + 0.4.*pred_labels_3);
end

