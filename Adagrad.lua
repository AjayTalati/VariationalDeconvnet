function adaGradInit(data, opfunc, batchSize, adaGradInitRounds)
    local h = {}
    for i = 1, batchSize*adaGradInitRounds+1, batchSize do
        local batch = data[{{i,i+batchSize-1}}]

        local weights, grads, lowerbound = opfunc(batch)

        for j=1,#grads do
		grads[j] = grads[j]:double()
            if h[j] == nil then
                h[j] = torch.cmul(grads[j],grads[j]):add(0.01)
            else
                h[j]:add(torch.cmul(grads[j],grads[j]))
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
	grads[i] = grads[i]:double()
        h[i]:add(torch.cmul(grads[i],grads[i]))

        local prior = 0
        if i % 2 ~= 0 then
            prior = -torch.mul(weights[i]:double(),0.5):mul(batchSize/N)
        end

        local update = torch.Tensor(h[i]:size()):fill(learningRate)
        update:cdiv(torch.sqrt(h[i])):cmul(torch.add(grads[i],prior))
    end

    collectgarbage()
    return lowerbound
end
