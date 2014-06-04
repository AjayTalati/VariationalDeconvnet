local LinearCR, parent = torch.class('nn.LinearCR', 'nn.Linear')

--Custom reset function
function LinearCR:__init(inputSize, outputSize)
    parent.__init(self, inputSize, outputSize)
    self:reset()
end

function LinearCR:reset()
    sigmaInit = 0.01
    self.weight:normal(0, 0.01)
    self.bias:normal(0, 0.01)
end
