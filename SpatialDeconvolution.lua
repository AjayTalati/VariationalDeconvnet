local SpatialDeconvolution, parent = torch.class('nn.SpatialDeconvolution', 'nn.Module')

function SpatialDeconvolution:__init(inputSize, outputSize,factor)
   parent.__init(self)

   self.weight = torch.Tensor(outputSize*factor*factor, inputSize)
   self.gradWeight = torch.Tensor(outputSize*factor*factor, inputSize)
   self.factor = factor
   
   self:reset()
end

function SpatialDeconvolution:reset()
    sigmaInit = 0.01
    self.weight:normal(0, 0.01)
    self.bias:normal(0, 0.01)
end


function SpatialDeconvolution:updateOutput(input)
   local nframe = input:size(1)
   local nunit = self.weight:size(1)

   self.output:resize(nframe, nunit)
   self.output:mm(input, self.weight:t())

   return self.output
end

function SpatialDeconvolution:updateGradInput(input, gradOutput)
    self.gradInput:mm(gradOutput, self.weight)
    return self.gradInput
end

function SpatialDeconvolution:accGradParameters(input, gradOutput, scale)
   scale = scale or 1

   local nframe = input:size(1)
   local nunit = self.bias:size(1)

   self.gradWeight:addmm(scale, gradOutput:t(), input)
end
