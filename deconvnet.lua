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
require 'KLDCriterion'
require 'BCECriterion'

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

BCE = nn.BCECriterion()
KLD = nn.KLDCriterion()

function display(input, reconstruction)
  gfx.image({input:reshape(3,32,32), reconstruction:reshape(3,32,32)}, {zoom=9, legends={'Input', 'Reconstruction'}})
end

opfunc = function(batch) 
    model:zeroGradParameters()

    f = model:forward(batch)

    local target = batch:reshape(100,3072)

    err = BCE:forward(f, target)
    df_dw = BCE:backward(f, target)

    model:backward(batch,df_dw)

    KLDerr = KLD:forward(model:get(1).output, target)
    dKLD_dw = KLD:backward(model:get(1).output, target)
    encoder:backward(batch,dKLD_dw)

    lowerbound = err  + KLDerr
    weights, grads = model:parameters()


    return weights, grads, lowerbound
end

local trsize = 50000
local tesize = 10000


-- load dataset
trainData = {
   data = torch.Tensor(trsize, 3072),
   labels = torch.Tensor(trsize),
   size = function() return trsize end
}

for i = 0,4 do
  subset = torch.load('cifar-10-batches-t7/data_batch_' .. (i+1) .. '.t7', 'ascii')
  trainData.data[{ {i*10000+1, (i+1)*10000} }] = subset.data:t()
  trainData.labels[{ {i*10000+1, (i+1)*10000} }] = subset.labels
end

-- trainData.data = trainData.data:double()

trainData.labels = trainData.labels + 1

subset = torch.load('cifar-10-batches-t7/test_batch.t7', 'ascii')
testData = {
   data = subset.data:t():double(),
   labels = subset.labels[1]:double(),
   size = function() return tesize end
}
testData.labels = testData.labels + 1

-- reshape data
trainData.data = trainData.data:div(255):reshape(trsize,3,32,32)
testData.data = testData.data:div(255):reshape(tesize,3,32,32)

epoch = 0


adaGradInitRounds = 5
h = adaGradInit(trainData.data, opfunc, adaGradInitRounds)
lowerboundlist = {}

while true do
    epoch = epoch + 1
    local lowerbound = 0
    local time = sys.clock()
    local shuffle = torch.randperm(trainData.data:size(1))
    local N = trainData.data:size(1)

    for i = 1, N, batchSize do
        local iend = math.min(N,i+batchSize-1)
        xlua.progress(iend, N)

        batch = torch.Tensor(iend-i+1,trainData.data:size(2),32,32)

        local k = 1
        for j = i,iend do
            batch[k] = trainData.data[shuffle[j]]:clone() 
            k = k + 1
        end

        batchlowerbound = adaGradUpdate(batch, opfunc, h)
        lowerbound = lowerbound + batchlowerbound
    end
    print("Epoch: " .. epoch .. " Lowerbound: " .. lowerbound/N .. " time: " .. sys.clock() - time)
    table.insert(lowerboundlist, lowerbound/N)

    -- if epoch % 2 == 0 and epoch ~= 0 then
    --   display(batch[{{1},{},{},{}}], f[{{1},{}}])
    -- end

    if epoch % 5 == 0 and epoch ~= 0 then
        print("Saving weights...")
        weights, bias = model:getParameters()
        torch.save('params/' .. epoch .. '_weight.t7', weights)
        torch.save('params/' .. epoch .. '_bias.t7', bias)
        torch.save('params/' .. epoch .. '_adagrad.t7', h)
        torch.save('params/lowerbound.t7', torch.Tensor(lowerboundlist))
    end
end