require 'sys'
require 'xlua'
require 'torch'
require 'nn'
require 'optim'


require 'KLDCriterion'

require 'LinearCR'
require 'Reparametrize'
require 'SpatialDeconvolution'

require 'load'

dname,fname = sys.fpath()
cmd = torch.CmdLine()
cmd:text()
cmd:text('Deconvolutional network')
cmd:text()
cmd:text('Options:')
cmd:option('-save', fname:gsub('.lua',''), 'subdirectory to save/log experiments in')
cmd:option('-continue', false, 'load parameters from earlier training')
cmd:option('-verbose', false, 'add verbosity, loooots of prints')
cmd:option('-cuda', false, 'use CUDA modules')

cmd:text()
opt = cmd:parse(arg)

torch.manualSeed(1)

---Required 
batchSize = 128 

require (opt.save .. '/config')

if continuous then
    criterion = nn.GaussianCriterion()
else
    criterion = nn.BCECriterion()
    criterion.sizeAverage = false
end

KLD = nn.KLDCriterion()
KLD.sizeAverage = false

if opt.cuda then
    criterion:cuda()
    KLD:cuda()
    model:cuda()
end


parameters, gradients = model:getParameters()

config = {
    learningRate = -0.01
}

function getLowerbound(data)
    local lowerbound = 0
    N_data = data:size(1) - (data:size(1) % batchSize)
    for i = 1, N_data, batchSize do
        local batch = data[{{i,i+batchSize-1},{}}]
        local f = model:forward(batch)
        local target = batch:reshape(batchSize,total_output_size)
        local err = - criterion:forward(f, target)

        local encoder_output = model:get(1).output

        local KLDerr = KLD:forward(encoder_output, target)

        lowerbound = lowerbound + err + KLDerr
    end
    return lowerbound
end


--Currently not working with new Adagrad
if opt.continue == true then 
    print("Loading old weights!")
    lowerboundlist = torch.load(opt.save ..        'lowerbound.t7')
    lowerbound_test_list =  torch.load(opt.save .. 'lowerbound_test.t7')
    h = torch.load(opt.save .. 'adagrad.t7')
    w = torch.load(opt.save .. 'weights.t7')

    weights, gradients = model:getParameters()

    weights:copy(w)
    if opt.verbose then
        print(weights:size())
    end
    epoch = lowerboundlist:size(1)
else
    epoch = 0
    state = {}
end

while true do
    epoch = epoch + 1
    local lowerbound = 0
    local time = sys.clock()
    local shuffle = torch.randperm(trainData.data:size(1))

    --Make sure batches are always batchSize
    local N = trainData.data:size(1) - (trainData.data:size(1) % batchSize)
    local N_test = testData.data:size(1) - (testData.data:size(1) % batchSize)

    for i = 1, N, batchSize do
        xlua.progress(i+batchSize-1, N)

        --Prepare Batch
        local batch = torch.Tensor(batchSize,colorchannels,input_size,input_size)

         if opt.cuda then
            batch = batch:cuda()
        end 

        local k = 1

        for j = i,i+batchSize-1 do
            batch[k] = trainData.data[shuffle[j]]:clone() 
            k = k + 1
        end

        --Optimization function
        local opfunc = function(x)
            if x ~= parameters then
                parameters:copy(x)
            end

            model:zeroGradParameters()
            local f = model:forward(batch)

            local target = target or batch.new()
            target:resizeAs(f):copy(batch)

            local err = - criterion:forward(f, target)
            local df_dw = criterion:backward(f, target):mul(-1)

            model:backward(batch,df_dw)
            local encoder_output = model:get(1).output

            local KLDerr = KLD:forward(encoder_output, target)
            local dKLD_dw = KLD:backward(encoder_output, target)

            encoder:backward(batch,dKLD_dw)

            local lowerbound = err  + KLDerr

            if opt.verbose then
                print("BCE",err/batch:size(1))
                print("KLD", KLDerr/batch:size(1))
                print("lowerbound", lowerbound/batch:size(1))
            end

            return lowerbound, gradients 
        end

        x, batchlowerbound = optim.adagrad(opfunc, parameters, config, state)

        lowerbound = lowerbound + batchlowerbound[1]
    end

    print("Epoch: " .. epoch .. " Lowerbound: " .. lowerbound/N .. " time: " .. sys.clock() - time)

    --Keep track of the lowerbound over time
    if lowerboundlist then
        lowerboundlist = torch.cat(lowerboundlist,torch.Tensor(1,1):fill(lowerbound/N),1)
    else
        lowerboundlist = torch.Tensor(1,1):fill(lowerbound/N)
    end


    --Compute the lowerbound of the test set and save it
    if epoch % 1 == 0 then
        lowerbound_test = getLowerbound(testData.data)

         if lowerbound_test_list then
            lowerbound_test_list = torch.cat(lowerbound_test_list,torch.Tensor(1,1):fill(lowerbound_test/N_test),1)
        else
            lowerbound_test_list = torch.Tensor(1,1):fill(lowerbound_test/N_test)
        end

        print('testlowerbound = ' .. lowerbound_test/N_test)

        --Save everything to be able to restart later
        torch.save(opt.save .. '/model', model)
        torch.save(opt.save .. '/weights.t7', weights)
        torch.save(opt.save .. '/adagrad.t7', state)
        torch.save(opt.save .. '/lowerbound.t7', torch.Tensor(lowerboundlist))
        torch.save(opt.save .. '/lowerbound_test.t7', torch.Tensor(lowerbound_test_list))
    end
end
