function adaGradInit(data, opfunc, batchSize, adaGradInitRounds)
    local h = {}
    for i = 1, batchSize*adaGradInitRounds+1, batchSize do
        local batch = data[{{i,i+batchSize-1}}]

        local weights, grads, lowerbound = opfunc(batch)

        for j=1,#grads do            
            if h[j] == nil then
                h[j] = torch.Tensor():resizeAs(grads[j]):copy(grad[j])
                h[j] = torch:(grads[j]):add(0.01)
            else
                h[j]:add(grads[j]:pow(2))
            end
        end
        collectgarbage()
    end

    return h
end

function adaGradUpdate(batch, N, learningRate, opfunc, h)
    local batchSize = batch:size(1)
    local weights, grads, lowerbound = opfunc(batch)

    for i=1,#h do

        local prior = 0
        if i % 2 ~= 0 then
            prior = torch.Tensor():resizeAs(weights[i]):fill(0.5)
            prior:mul(weights[i]):mul(batchSize/N)
        end

        local update = torch.Tensor(h[i]:size()):fill(learningRate)

        update:cdiv(torch.sqrt(h[i])):cmul(prior:add(grads[i]))


        weights[i]:add(update)

        h[i]:add(grads[i]:pow(2))

    end


    collectgarbage()
    return lowerbound
end
