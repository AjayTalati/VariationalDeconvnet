local SpatialDeconvolution, parent = torch.class('nn.SpatialDeconvolution', 'nn.Module')

function SpatialDeconvolution:__init(inputSize, outputSize,factor)
   parent.__init(self)

   self.weight = torch.Tensor(outputSize*factor*factor, inputSize)
   self.gradWeight = torch.Tensor(outputSize*factor*factor, inputSize)
   self.factor = factor
   
   self:reset()
end

function SpatialDeconvolution:reset()
    local sigmaInit = 0.01
    self.weight:normal(0, sigmaInit)
end


function SpatialDeconvolution:updateOutput(input)
  self.output:resize(input:size(1), self.weight:size(1))
  self.output:mm(input, self.weight:t())

  print(self.output:size())

   return self.output
end

function SpatialDeconvolution:updateGradInput(input, gradOutput)
  self.gradInput = torch.mm(gradOutput, self.weight)
  return self.gradInput
end

function SpatialDeconvolution:accGradParameters(input, gradOutput, scale)
   local scale = scale or 1
   self.gradWeight:addmm(scale, gradOutput:t(), input)
   return self.gradWeight
end
