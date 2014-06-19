function adaGradInit(data, opfunc, batchSize, adaGradInitRounds)
    local h = {}
    for i = 1, batchSize*adaGradInitRounds+1, batchSize do
        local batch = data[{{i,i+batchSize-1}}]

        local weights, grads, lowerbound = opfunc(batch)

        for j=1,#grads do            
            if h[j] == nil then
                h[j] = grads[j]:clone()
                h[j]:cmul(grads[j]):add(0.01)
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

	print(grads)
    for i=1,#h do

        local prior = weights[i].new()
	prior:resizeAs(weights[i]):fill(0)

        if i % 2 == 1 then
            prior:add(-0.5):cmul(weights[i]):mul(batchSize/N)
	end

        local update = h[i].new()
	update:resizeAs(h[i]):fill(learningRate)

        update:cdiv(h[i]):cdiv(h[i]):cmul(prior:add(grads[i]))
	print(torch.norm(update:double()))

        weights[i]:add(update)

        h[i]:add(grads[i]:pow(2))
    end

    collectgarbage()
    return lowerbound
end
