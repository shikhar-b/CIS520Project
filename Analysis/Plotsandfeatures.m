data_points = size(train_inputs,1);
plotmatrix([train_inputs(:,1), train_inputs(:,2),train_inputs(:,3)],[train_labels(:,1),train_labels(:,2)],'','xos');