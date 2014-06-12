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

function LinearCR:updateOutput(input)
   if input:dim() == 1 then
      self.output:resize(self.bias:size(1))
      self.output:copy(self.bias)
      self.output:addmv(1, self.weight, input)
   elseif input:dim() == 2 then
      local nframe = input:size(1)
      local nunit = self.bias:size(1)

      self.output:resize(nframe, nunit)
      self.output:zero():addr(1, input.new(nframe):fill(1), self.bias)
      self.output:addmm(1, input, self.weight:t())
   else
      error('input must be vector or matrix')
   end

   return self.output
end