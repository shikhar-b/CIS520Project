function idxs = KM(inputs)
	rng(100);
	idxs = kmeans(inputs, 5);
