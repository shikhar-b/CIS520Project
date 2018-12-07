function [idxs, kc] = KM(inputs)
	rng(100);
	[idxs, kc] = kmeans(inputs, 5);
