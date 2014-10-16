
function optimx.rmsprop(opfunc, x, config, state)
   -- get parameters
   local fx,dfdx = opfunc(x)
   local config = config or {}
   local decay = config.decay or 0.9
   local learningRate = config.learningRate or 1e-3
   local maxLearningRate = config.maxLearningRate or 50
   local rms = config.rms or torch.abs(dfdx:clone())
    -- calculate update steps
   local dfdx_sq = dfdx:clone()
   local dfdx_sq = dfdx_sq:cmul(dfdx_sq)
   rms:mul(decay):add(dfdx_sq:mul(1-decay))
   local rms_sqrt = torch.sqrt(rms+1e-8)
   dfdx:cdiv(rms_sqrt)
   dfdx:apply(function (ss) if ss > maxLearningRate then return maxLearningRate end end)
   config.rms = rms
   x:add(-learningRate, dfdx)
   -- return x*, f(x) before optimization
   return x,{fx}
end
