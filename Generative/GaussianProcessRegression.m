numtrPoints = size(train_inputs,1);
train_data_1 = [train_inputs train_labels(:,2:9)];
train_labels_1 = train_labels(:,1);
train_data_2 = [train_inputs train_labels(:,1),train_labels(:,3:9)];
train_labels_2 = train_labels(:,2);
train_data_3 = [train_inputs train_labels(:,1:2),train_labels(:,4:9)];
train_labels_3 = train_labels(:,3);
train_data_4 = [train_inputs train_labels(:,1:3),train_labels(:,5:9)];
train_labels_4 = train_labels(:,4);
train_data_5 = [train_inputs train_labels(:,1:4),train_labels(:,6:9)];
train_labels_5 = train_labels(:,5);
train_data_6 = [train_inputs train_labels(:,1:5),train_labels(:,7:9)];
train_labels_6 = train_labels(:,6);
train_data_7 = [train_inputs train_labels(:,1:6),train_labels(:,8:9)];
train_labels_7 = train_labels(:,7);
train_data_8 = [train_inputs train_labels(:,1:7),train_labels(:,9)];
train_labels_8 = train_labels(:,8);
train_data_9 = [train_inputs train_labels(:,1:8)];
train_labels_9 = train_labels(:,9);

[trainInd, valInd, testInd] = divideblock(numtrPoints, 0.7, 0.2, 0.1);

training_labels = train_labels(trainInd,:);
val_labels = train_labels(valInd,:);
test_labels = train_labels(testInd,:);

training_data_1 = train_data_1(trainInd,:);
validation_data_1 = train_data_1(valInd,:);
testing_data_1 = train_data_1(testInd,:);
training_labels_1 = train_labels_1(trainInd,:);
validation_labels_1 = train_labels_1(valInd,:);
testing_labels_1 = train_labels_1(testInd,:);
training_data_2 = train_data_2(trainInd,:);
validation_data_2 = train_data_2(valInd,:);
testing_data_2 = train_data_2(testInd,:);
training_labels_2 = train_labels_2(trainInd,:);
validation_labels_2 = train_labels_2(valInd,:);
testing_labels_2 = train_labels_2(testInd,:);
training_data_3 = train_data_3(trainInd,:);
validation_data_3 = train_data_3(valInd,:);
testing_data_3 = train_data_3(testInd,:);
training_labels_3 = train_labels_3(trainInd,:);
validation_labels_3 = train_labels_3(valInd,:);
testing_labels_3 = train_labels_3(testInd,:);
training_data_4 = train_data_4(trainInd,:);
validation_data_4 = train_data_4(valInd,:);
testing_data_4 = train_data_4(testInd,:);
training_labels_4 = train_labels_4(trainInd,:);
validation_labels_4 = train_labels_4(valInd,:);
testing_labels_4 = train_labels_4(testInd,:);
training_data_5 = train_data_5(trainInd,:);
validation_data_5 = train_data_5(valInd,:);
testing_data_5 = train_data_5(testInd,:);
training_labels_5 = train_labels_5(trainInd,:);
validation_labels_5 = train_labels_5(valInd,:);
testing_labels_5 = train_labels_5(testInd,:);
training_data_6 = train_data_6(trainInd,:);
validation_data_6 = train_data_6(valInd,:);
testing_data_6 = train_data_6(testInd,:);
training_labels_6 = train_labels_6(trainInd,:);
validation_labels_6 = train_labels_6(valInd,:);
testing_labels_6 = train_labels_6(testInd,:);
training_data_7 = train_data_7(trainInd,:);
validation_data_7 = train_data_7(valInd,:);
testing_data_7 = train_data_7(testInd,:);
training_labels_7 = train_labels_7(trainInd,:);
validation_labels_7 = train_labels_7(valInd,:);
testing_labels_7 = train_labels_7(testInd,:);
training_data_8 = train_data_8(trainInd,:);
validation_data_8 = train_data_8(valInd,:);
testing_data_8 = train_data_8(testInd,:);
training_labels_8 = train_labels_8(trainInd,:);
validation_labels_8 = train_labels_8(valInd,:);
testing_labels_8 = train_labels_8(testInd,:);
training_data_9 = train_data_9(trainInd,:);
validation_data_9 = train_data_9(valInd,:);
testing_data_9 = train_data_9(testInd,:);
training_labels_9 = train_labels_9(trainInd,:);
validation_labels_9 = train_labels_9(valInd,:);
testing_labels_9 = train_labels_9(testInd,:);

gprMdl1 = fitrgp(training_data_1, training_labels_1);
gprMdl2 = fitrgp(training_data_2, training_labels_2);
gprMdl3 = fitrgp(training_data_3, training_labels_3);
gprMdl4 = fitrgp(training_data_4, training_labels_4);
gprMdl5 = fitrgp(training_data_5, training_labels_5);
gprMdl6 = fitrgp(training_data_6, training_labels_6);
gprMdl7 = fitrgp(training_data_7, training_labels_7);
gprMdl8 = fitrgp(training_data_8, training_labels_8);
gprMdl9 = fitrgp(training_data_9, training_labels_9);

ypred1_training = resubPredict(gprMdl1);
ypred2_training = resubPredict(gprMdl2);
ypred3_training = resubPredict(gprMdl3);
ypred4_training = resubPredict(gprMdl4);
ypred5_training = resubPredict(gprMdl5);
ypred6_training = resubPredict(gprMdl6);
ypred7_training = resubPredict(gprMdl7);
ypred8_training = resubPredict(gprMdl8);
ypred9_training = resubPredict(gprMdl9);

ypred1_validation = predict(gprMdl1,validation_data_1);
ypred2_validation = predict(gprMdl2,validation_data_2);
ypred3_validation = predict(gprMdl3,validation_data_3);
ypred4_validation = predict(gprMdl4,validation_data_4);
ypred5_validation = predict(gprMdl5,validation_data_5);
ypred6_validation = predict(gprMdl6,validation_data_6);
ypred7_validation = predict(gprMdl7,validation_data_7);
ypred8_validation = predict(gprMdl8,validation_data_8);
ypred9_validation = predict(gprMdl9,validation_data_9);

ypred1_testing = predict(gprMdl1,testing_data_1);
ypred2_testing = predict(gprMdl2,testing_data_2);
ypred3_testing = predict(gprMdl3,testing_data_3);
ypred4_testing = predict(gprMdl4,testing_data_4);
ypred5_testing = predict(gprMdl5,testing_data_5);
ypred6_testing = predict(gprMdl6,testing_data_6);
ypred7_testing = predict(gprMdl7,testing_data_7);
ypred8_testing = predict(gprMdl8,testing_data_8);
ypred9_testing = predict(gprMdl9,testing_data_9);

ypred_training = [ypred1_training ypred2_training ypred3_training ypred4_training ypred5_training ypred6_training ypred7_training ypred8_training ypred9_training];
ypred_validation = [ypred1_validation ypred2_validation ypred3_validation ypred4_validation ypred5_validation ypred6_validation ypred7_validation ypred8_validation ypred9_validation];
ypred_testing = [ypred1_testing ypred2_testing ypred3_testing ypred4_testing ypred5_testing ypred6_testing ypred7_testing ypred8_testing ypred9_testing];
training_error = error_metric(ypred_training, training_labels);
validation_error = error_metric(ypred_validation, val_labels);
testing_error = error_metric(ypred_testing, test_labels);