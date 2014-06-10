-- load dataset
function loadCifar(trsize,tesize)
	local trainData = {
	   data = torch.Tensor(trsize, 3072),
	   labels = torch.Tensor(trsize),
	   size = function() return trsize end
	}

	for i = 0,4 do
	  local subset = torch.load('cifar-10-batches-t7/data_batch_' .. (i+1) .. '.t7', 'ascii')
	  trainData.data[{ {i*10000+1, (i+1)*10000} }] = subset.data:t()
	  trainData.labels[{ {i*10000+1, (i+1)*10000} }] = subset.labels
	end

	-- trainData.data = trainData.data:double()

	trainData.labels = trainData.labels + 1

	local subset = torch.load('cifar-10-batches-t7/test_batch.t7', 'ascii')
	local testData = {
	   data = subset.data:t():double(),
	   labels = subset.labels[1]:double(),
	   size = function() return tesize end
	}
	testData.labels = testData.labels + 1

	-- reshape data
	trainData.data = trainData.data:div(255):reshape(trsize,3,32,32)
	testData.data = testData.data:div(255):reshape(tesize,3,32,32)

	return trainData,testData
end



--trainData.data = trainData.data[{{},{},{2,31},{2,31}}]
--testData.data = testData.data[{{},{},{2,31},{2,31}}]