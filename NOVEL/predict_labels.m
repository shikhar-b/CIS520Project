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
addpath('Random');
pred_labels_1 = ensembles1(train_inputs, train_labels, test_inputs);
pred_labels_2 = ensembles2(train_inputs, train_labels, test_inputs);
pred_labels_3 = GPRResidual(train_inputs, train_labels, test_inputs);
pred_labels = 0.4*pred_labels_1+0.4*pred_labels_2+0.2*pred_labels_3;
end

