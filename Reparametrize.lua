-- Based on JoinTable module

require 'nn'

local Reparametrize, parent = torch.class('nn.Reparametrize', 'nn.Module')

function Reparametrize:__init(dimension)
    parent.__init(self)
    self.size = torch.LongStorage()
    self.dimension = dimension
    self.gradInput = {}
end 

function Reparametrize:updateOutput(input)
    if torch.typename(input[1]) == 'torch.CudaTensor' then
        self.eps = torch.CudaTensor(input[2]:size(1),self.dimension)
    else
        self.eps = torch.Tensor(input[2]:size(1),self.dimension)
    end

    torch.randn(self.eps,input[2]:size(1),self.dimension)
    self.output = torch.mul(input[2],0.5):exp():cmul(self.eps)

    -- Add the mean_
    self.output:add(input[1])

    return self.output
end

function Reparametrize:updateGradInput(input, gradOutput)
    -- Derivative with respect to mean is 1
    self.gradInput[1] = gradOutput:clone()
    
    --test gradient with Jacobian
    self.gradInput[2] = torch.mul(input[2],0.5):exp():mul(0.5):cmul(self.eps)
    self.gradInput[2]:cmul(gradOutput)

    return self.gradInput
end
