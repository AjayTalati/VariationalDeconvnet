require 'torch'
require 'nn'

require 'linearCR'
require 'Reparametrize'

require 'Adagrad'

------------------------------------------------------------
-- convolutional network
------------------------------------------------------------
-- Grey Images
-- stage 1 : 3 input channels, 10 output, 5x5 filter, 2x2 stride
local filter_size = 4
local stride = 2
local dim_hidden = 100
local input_size = 32


encoder = nn.Sequential()
encoder:add(nn.SpatialConvolution(3,10,filter_size,filter_size,stride,stride))
encoder:add(nn.SoftPlus())
encoder:add(nn.Reshape(10*5*5))

z = nn.ConcatTable()
z:add(nn.LinearCR(10*5*5, dim_hidden))
z:add(nn.LinearCR(10*5*5, dim_hidden))

encoder:add(z)

decoder = nn.Sequential()
decoder:add(nn.LinearCR(dim_hidden, 10*5*5))
decoder:add(nn.Reshape(25,10))
encoder:add(nn.SpatialConvolution(10,3,filter_size,filter_size,stride,stride))

decoder:add()

model = nn.Sequential()
model:add(encoder)
model:add(nn.Reparametrize())
model:add(decoder)

Gaussian = nn.GaussianCriterion()
KLD = nn.KLDCriterion()

opfunc = function(batch) 
    model:zeroGradParameters()

    f = model:forward(batch)
    err = Gaussian:forward(f, batch)
    df_dw = Gaussian:backward(f, batch)
    model:backward(batch,df_dw)

    KLDerr = KLD:forward(model:get(1).output, batch)
    dKLD_dw = KLD:backward(model:get(1).output, batch)
    encoder:backward(batch,dKLD_dw)

    lowerbound = err  + KLDerr
    weights, grads = model:parameters()


    return weights, grads, lowerbound
end

-- load dataset
trainData = {
   data = torch.Tensor(50000, 3072),
   labels = torch.Tensor(50000),
   size = function() return trsize end
}

for i = 0,4 do
   subset = torch.load('cifar-10-batches-t7/data_batch_' .. (i+1) .. '.t7', 'ascii')
   trainData.data[{ {i*10000+1, (i+1)*10000} }] = subset.data:t()
   trainData.labels[{ {i*10000+1, (i+1)*10000} }] = subset.labels
end

trainData.labels = trainData.labels + 1

subset = torch.load('cifar-10-batches-t7/test_batch.t7', 'ascii')
testData = {
   data = subset.data:t():double(),
   labels = subset.labels[1]:double(),
   size = function() return tesize end
}
testData.labels = testData.labels + 1

-- reshape data
trainData.data = trainData.data:reshape(trsize,3,32,32)
testData.data = testData.data:reshape(tesize,3,32,32)

while true do
    epoch = epoch + 1
    local lowerbound = 0
    local time = sys.clock()
    local shuffle = torch.randperm(trainData.data:size(1))
    local N = trainData.data:size(1)

    for i = 1, N, batchSize do
        local iend = math.min(N,i+batchSize-1)
        -- xlua.progress(iend, N)

        local batch = torch.Tensor(iend-i+1,data.train:size(2))

        local k = 1
        for j = i,iend do
            batch[k] = trainData.data[shuffle[j]]:clone() 
            k = k + 1
        end

        batchlowerbound = adaGradUpdate(batch, opfunc, h)
        lowerbound = lowerbound + batchlowerbound
    end
