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
end


function SpatialDeconvolution:updateOutput(input)
   local batchsize = input:size(1)

   input = input:reshape(input:size(1) * input:size(2),input:size(3))

   self.output:resize(input:size(1), self.weight:size(1))

   print(input)
   print(self.weight:t())
   self.output:mm(input, self.weight:t())

   -- self.output = self.output:resize(100,64,48)
    -- self.output = self.output:resize(6400,4,4,3)
    -- self.output = self.output:resize(25600,4,3)
    -- self.output = self.output:resize(3200,32,3)
    -- self.output = self.output:resize(100,32,32,3)
    self.output = self.output:resize(100,3,32,32)
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
