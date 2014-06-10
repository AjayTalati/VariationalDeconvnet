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
require 'SpatialZeroPaddingC'

require 'load'

------------------------------------------------------------
-- deconvolutional network
------------------------------------------------------------
-- torch.setnumthreads(2)
-- print('<torch> set nb of threads to ' .. torch.getnumthreads())


require 'config/1-layer-nopadding'

torch.save('params/model',model)

BCE = nn.BCECriterion()
KLD = nn.KLDCriterion()

opfunc = function(batch) 
    model:zeroGradParameters()
    local f = model:forward(batch)
    local target = batch:reshape(100,total_output_size)
    local err = BCE:forward(f, target)
    local df_dw = BCE:backward(f, target)

    model:backward(batch,df_dw)

    local KLDerr = KLD:forward(model:get(1).output, target)
    local dKLD_dw = KLD:backward(model:get(1).output, target)
    encoder:backward(batch,dKLD_dw)

    local lowerbound = err  + KLDerr
    local weights, grads = model:parameters()


    return weights, grads, lowerbound
end

<<<<<<< HEAD:deconvnet.lua
local trsize = 50000
local tesize = 10000


-- load dataset
local trainData = {
   data = torch.Tensor(trsize, 3072),
   labels = torch.Tensor(trsize),
   size = function() return trsize end
}

for i = 0,0 do
  subset = torch.load('cifar-10-batches-t7/data_batch_' .. (i+1) .. '.t7', 'ascii')
  trainData.data[{ {i*10000+1, (i+1)*10000} }] = subset.data:t()
  trainData.labels[{ {i*10000+1, (i+1)*10000} }] = subset.labels
end

-- trainData.data = trainData.data:double()

trainData.labels = trainData.labels + 1

subset = torch.load('cifar-10-batches-t7/test_batch.t7', 'ascii')
local testData = {
   data = subset.data:t():double(),
   labels = subset.labels[1]:double(),
   size = function() return tesize end
}
testData.labels = testData.labels + 1

-- reshape data
trainData.data = trainData.data:div(255):reshape(trsize,3,32,32)
--trainData.data = trainData.data[{{},{},{2,31},{2,31}}]
testData.data = testData.data:div(255):reshape(tesize,3,32,32)
--testData.data = testData.data[{{},{},{2,31},{2,31}}]
=======
trsize = 50000
tesize = 10000
>>>>>>> 6c34e17e1d736dc12615b8c4f9630853a0cb1a4d:main.lua

trainData, testData = loadCifar(trsize,tesize)

epoch = 0

adaGradInitRounds = 2
h = adaGradInit(trainData.data, opfunc, batchSize, adaGradInitRounds)
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

        local batch = torch.Tensor(iend-i+1,trainData.data:size(2),input_size,input_size)

        local k = 1
        for j = i,iend do
            batch[k] = trainData.data[shuffle[j]]:clone() 
            k = k + 1
        end

        batchlowerbound = adaGradUpdate(batch,N, learningRate, opfunc, h)
        lowerbound = lowerbound + batchlowerbound
    end
    print("Epoch: " .. epoch .. " Lowerbound: " .. lowerbound/N .. " time: " .. sys.clock() - time)
    table.insert(lowerboundlist, lowerbound/N)

    if epoch % 5 == 0 and epoch ~= 0 then
        print("Saving weights...")
        weights, gradients = model:getParameters()
        torch.save('params/' .. epoch .. '_weights.t7', weights)
        torch.save('params/' .. epoch .. '_adagrad.t7', h)
        torch.save('params/lowerbound.t7', torch.Tensor(lowerboundlist))
    end
end