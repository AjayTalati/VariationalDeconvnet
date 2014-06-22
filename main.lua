require 'sys'
require 'xlua'
require 'torch'
require 'nn'

require 'AdagradCUDA'
require 'KLDCriterion'

require 'LinearCR'
require 'Reparametrize'
require 'SpatialDeconvolution'
require 'SpatialZeroPaddingC'

require 'load'

dname,fname = sys.fpath()
cmd = torch.CmdLine()
cmd:text()
cmd:text('Deconvolutional network')
cmd:text()
cmd:text('Options:')
cmd:option('-save', fname:gsub('.lua',''), 'subdirectory to save/log experiments in')
cmd:option('-continue', false, 'load parameters from earlier training')
cmd:option('-seed', 'yes', 'fixed input seed for repeatable experiments')


cmd:text()
opt = cmd:parse(arg)

if opt.seed == 'yes' then
    torch.manualSeed(1)
end

require (opt.save .. '/config')

BCE = nn.BCECriterion()
BCE.sizeAverage = false
KLD = nn.KLDCriterion()

opfunc = function(batch) 
    model:zeroGradParameters()
    local f = model:forward(batch)
    -- local target = batch[{{},{},{3,34},{3,34}}]:reshape(100,total_output_size)
    local target = batch:double():reshape(batchSize,total_output_size)
    local err = BCE:forward(f, target)
    local df_dw = BCE:backward(f, target)

    model:backward(batch,df_dw)
    local encoder_output = model:get(1).output

    if cuda then
       encoder_output[1] = encoder_output[1]:double()
       encoder_output[2] = encoder_output[2]:double()
    end

    local KLDerr = KLD:forward(encoder_output, target)
    local dKLD_dw = KLD:backward(encoder_output, target)

    if cuda then
        dKLD_dw[1] = dKLD_dw[1]:cuda()
        dKLD_dw[2] = dKLD_dw[2]:cuda()
    end

    encoder:backward(batch,dKLD_dw)

    local lowerbound = err  + KLDerr
    local weights, grads = model:parameters()

    return weights, grads, lowerbound
end

function getLowerbound(data)
    local lowerbound = 0
    for i = 1, data:size(1), batchSize do
        local iend = math.min(data:size(1),i+batchSize-1)
        xlua.progress(iend, data:size(1))

        local batch = data[{{i,iend},{}}]
        local f = model:forward(batch)
        local target = batch:double():reshape(batchSize,total_output_size)
        local err = BCE:forward(f, target)

        local encoder_output = model:get(1).output
        
        if cuda then
            encoder_output[1] = encoder_output[1]:double()
            encoder_output[2] = encoder_output[2]:double()
        end

        local KLDerr = KLD:forward(encoder_output, target)

        lowerbound = lowerbound + err + KLDerr
    end
    return lowerbound
end


if opt.continue == true then --NB need to convert tensor to list!
    lowerboundlist = torch.load(opt.load ..        'lowerbound.t7')
    lowerbound_test_list =  torch.load(opt.load .. 'lowerbound_test.t7')
    h = torch.load(opt.load .. 'adagrad.t7')
    epoch = lowerboundList:size(1)
else
    epoch = 0
    lowerboundlist = {}
    lowerbound_test_list = {}
    h = adaGradInit(trainData.data, opfunc, batchSize, initrounds)
end

while true do
    epoch = epoch + 1
    local lowerbound = 0
    local time = sys.clock()
    local shuffle = torch.randperm(trainData.data:size(1))
    local N = trainData.data:size(1)
    local N_test = testData.data:size(1)

    for i = 1, N, batchSize do
        local iend = math.min(N,i+batchSize-1)
        xlua.progress(iend, N)

        local batch = torch.Tensor(iend-i+1,trainData.data:size(2),input_size,input_size)

        if cuda then
            batch = torch.CudaTensor(iend-i+1,trainData.data:size(2),input_size,input_size)
        end

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

    if epoch % 5 == 0 then
        print('Calculating test lowerbound\n')
        lowerbound_test = getLowerbound(testData.data)
        table.insert(lowerbound_test_list, lowerbound_test/N_test)
        print('testlowerbound = ')
        print(lowerbound_test/N_test)
        print("Saving weights...")
        weights, gradients = model:getParameters()

        torch.save(opt.save .. '/model', model)
        torch.save(opt.save .. '/weights.t7', weights)
        torch.save(opt.save .. '/adagrad.t7', h)
        torch.save(opt.save .. '/lowerbound.t7', torch.Tensor(lowerboundlist))
        torch.save(opt.save .. '/lowerbound_test.t7', torch.Tensor(lowerbound_test_list))
    end
end
