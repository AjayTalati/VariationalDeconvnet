require 'sys'
require 'xlua'
require 'torch'
require 'nn'

-- Remember to start server!
-- luajit -lgfx.start
-- luajit -lgfx.stop
gfx = require 'gfx.js'


require 'LinearCR'
require 'Reparametrize'
require 'Adagrad'
require 'SpatialDeconvolution'


------------------------------------------------------------
-- deconvolutional network
------------------------------------------------------------
-- torch.setnumthreads(2)
-- print('<torch> set nb of threads to ' .. torch.getnumthreads())

local filter_size = 4
local stride = 4
local dim_hidden = 100
local input_size = 32

-- NOT GENERIC 
local map_size = (input_size / stride) ^ 2
local feature_maps = 10

batchSize = 100
learningRate = 0.05

encoder = nn.Sequential()
encoder:add(nn.SpatialConvolution(3,feature_maps,filter_size,filter_size,stride,stride))
encoder:add(nn.Threshold(0,0))
encoder:add(nn.Reshape(feature_maps * map_size))

z = nn.ConcatTable()
z:add(nn.LinearCR(feature_maps * map_size, dim_hidden))
z:add(nn.LinearCR(feature_maps * map_size, dim_hidden))

encoder:add(z)

decoder = nn.Sequential()
decoder:add(nn.LinearCR(dim_hidden, feature_maps * map_size))
decoder:add(nn.Reshape(map_size*batchSize,feature_maps))
decoder:add(nn.SpatialDeconvolution(feature_maps,3,stride))
decoder:add(nn.Sigmoid())
decoder:add(nn.Reshape(batchSize,3072))

model = nn.Sequential()
model:add(encoder)
model:add(nn.Reparametrize(dim_hidden))
model:add(decoder)



function display(reconstruction, input)
  gfx.image({input:reshape(3,32,32), reconstruction:reshape(3,32,32)}, {zoom=9, legends={'Input', 'Reconstruction'}})
end

epoch = 22

weights, gradients = model:getParameters()
weights:copy(torch.load('params/' .. epoch .. '_weights.t7'))

local trsize = 10000

-- load dataset
trainData = {
   data = torch.Tensor(trsize, 3072),
   labels = torch.Tensor(trsize),
   size = function() return trsize end
}

subset = torch.load('cifar-10-batches-t7/data_batch_1.t7', 'ascii')
trainData.data[{ {1, 10000} }] = subset.data:t()
trainData.labels[{ {1, 10000} }] = subset.labels

-- reshape data
trainData.data = trainData.data:div(255):reshape(trsize,3,32,32)

local shuffle = torch.randperm(trainData.data:size(1))
local N = trainData.data:size(1)


batch = torch.Tensor(batchSize,trainData.data:size(2),32,32)

local k = 1
for j = 1,batchSize do
    batch[k] = trainData.data[shuffle[j]]:clone() 
    k = k + 1
end

f = model:forward(batch)


for i = 1,10 do
	input = f[{{i},{}}]
	target = batch[{{i},{},{},{}}]

	display(input,target)
end



