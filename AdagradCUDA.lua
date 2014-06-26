function adaGradInit(data, opfunc, batchSize, adaGradInitRounds)
    local h = {}
    print("Running AdaGrad init")
    for i = 1, batchSize*adaGradInitRounds+1, batchSize do
        local batch = data[{{i,i+batchSize-1}}]

        local weights, grads, lowerbound = opfunc(batch)

        for j=1,#grads do            
            if h[j] == nil then
                h[j] = grads[j]:clone()
                h[j]:cmul(grads[j]):add(1)
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
        local hupdate = grads[i]:clone()
        h[i]:add(hupdate:cmul(hupdate))

        local prior = weights[i].new()
    	prior:resizeAs(weights[i]):fill(0)


        if i % 2 == 1 then
            prior:add(-0.5):cmul(weights[i]):mul(batchSize/N)
    	end

        local update = h[i]:clone()
        update:pow(-0.5):mul(learningRate):cmul(prior:add(grads[i]))

        weights[i]:add(update)
    end


    collectgarbage()
    return lowerbound
end
