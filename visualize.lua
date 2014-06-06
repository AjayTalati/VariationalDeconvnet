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
require 'SpatialDeconvolution'


------------------------------------------------------------
-- deconvolutional network
------------------------------------------------------------
-- torch.setnumthreads(2)
-- print('<torch> set nb of threads to ' .. torch.getnumthreads()


function display(reconstruction, input)
  gfx.image({input:reshape(3,32,32), reconstruction:reshape(3,32,32)}, {zoom=9, legends={'Input', 'Reconstruction'}})
end

epoch = 42

model = torch.load('params/model')

weights, gradients = model:getParameters()
weights:copy(torch.load('params/' .. epoch .. '_weights.t7'))

featuremaps = weights[{{1,30*3*4*4}}]:reshape(30,3,4,4)

features = {}
for i=1,30 do
	table.insert(features,featuremaps[{{i},{1},{},{}}]:squeeze())
end

gfx.image(features,{zoom=20, legends = {'','','','','','','','','','','','','','','','','','','','','','','','','','','','','',''}})

features = {}
for i=1,30 do
	table.insert(features,featuremaps[{{i},{2},{},{}}]:squeeze())
end

gfx.image(features,{zoom=20, legends = {'','','','','','','','','','','','','','','','','','','','','','','','','','','','','',''}})

features = {}
for i=1,30 do
	table.insert(features,featuremaps[{{i},{3},{},{}}]:squeeze())
end

gfx.image(features,{zoom=20, legends = {'','','','','','','','','','','','','','','','','','','','','','','','','','','','','',''}})

io.read()

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



