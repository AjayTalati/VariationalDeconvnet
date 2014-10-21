--[[ An implementation of RMSprop

ARGS:

- 'opfunc' : a function that takes a single input (X), the point
             of a evaluation, and returns f(X) and df/dX
- 'x'      : the initial point
- 'config` : a table with configuration parameters for the optimizer
- 'config.learningRate'      : learning rate
- 'config.learningRateDecay' : learning rate decay
- 'config.weightDecay'       : weight decay
- 'config.momentum'          : momentum
- 'config.dampening'         : dampening for momentum
- 'config.nesterov'          : enables Nesterov momentum
- 'state'                    : a table describing the state of the optimizer; after each
                              call the state is modified
- 'state.rms'                 : vector of individual learning rates

RETURN:
- `x`     : the new x vector
- `f(x)`  : the function, evaluated before the update

]]

function rmsprop(opfunc, x, config, state)
   -- get parameters
   local config = config or {}
   local state = state or config
   local lr = config.learningRate or 1e-3
   -- local lrd = config.learningRateDecay or 0
   local b1 = config.momentumDecay or 1
   local b2 = config.updateDecay or 1

   local fx, dfdx = opfunc(x)

   state.evalCounter = state.evalCounter or 0
   state.m = state.m or torch.Tensor():resizeAs(dfdx):fill(0)
   state.v = state.v or torch.Tensor():resizeAs(dfdx):fill(0)
   
   -- Decay term
   state.m:mul(1 - b1)

   -- New term 
   state.momentum_dfdx = state.momentum_dfdx or torch.Tensor():typeAs(dfdx):resizeAs(dfdx)
   state.momentum_dfdx:copy(dfdx)

   state.m:add(state.momentum_dfdx:mul(b1))

   -- Decay term of update
   state.v:mul(1 - b2)

   -- New update
   dfdx:cmul(dfdx):mul(b2)

   state.v:add(dfdx)

   -- calculate update step
   state.evalCounter = state.evalCounter + 1

   --Might CUDA
   state.momentum_update = state.momentum_update or torch.Tensor():typeAs(state.m):resizeAs(state.m)
   state.momentum_update:copy(state.m)

   state.update = state.update or torch.Tensor():typeAs(state.v):resizeAs(state.v)
   state.update:copy(state.v)

   state.momentum_update:cdiv(state.update:add(1e-8):sqrt())

   -- No idea if and how this works on CUDA
   local gamma = (math.sqrt(1 - math.pow(1 - b2,state.evalCounter))/(1 - math.pow(1 - b1, state.evalCounter)))
   state.momentum_update:mul(gamma)

   x:add(-lr, state.momentum_update)

   -- return x*, f(x) before optimization
   return x,{fx}
end
