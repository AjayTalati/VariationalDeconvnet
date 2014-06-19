local SpatialDeconvolution, parent = torch.class('nn.SpatialDeconvolution', 'nn.Module')

function SpatialDeconvolution:__init(inputSize, outputSize,factor)
    parent.__init(self)

    self.weight = torch.Tensor(outputSize * factor * factor, inputSize)
    self.bias = torch.Tensor(outputSize * factor * factor)

    self.gradWeight = torch.Tensor(outputSize * factor * factor, inputSize)
    self.gradBias = torch.Tensor(outputSize * factor * factor)

   self.factor = factor
   
   self:reset()
end

function SpatialDeconvolution:reset()
    local sigmaInit = 0.01
    self.weight:normal(0, sigmaInit)
    self.bias:normal(0, sigmaInit)


end


function SpatialDeconvolution:updateOutput(input)
    if torch.typename(input) == 'torch.CudaTensor' then
        self.output = torch.CudaTensor()
    else
        self.output = torch.Tensor()
    end

    local nframe = input:size(1)
    local nunit = self.bias:size(1)

    self.output:resize(nframe, nunit)
    self.output:zero():addr(1, input.new(nframe):fill(1), self.bias)

    -- self.output:resize(input:size(1), self.weight:size(1)):fill(self.bias)

    self.output:addmm(input, self.weight:t())

   return self.output
end

function SpatialDeconvolution:updateGradInput(input, gradOutput)
    if torch.typename(input) == 'torch.CudaTensor' then
        self.gradInput = torch.CudaTensor()
    else
        self.gradInput = torch.Tensor()
    end

    self.gradInput:resize(gradOutput:size(1), self.weight:size(2)):fill(0)
    self.gradInput:addmm(gradOutput, self.weight)
    
    return self.gradInput
end

function SpatialDeconvolution:accGradParameters(input, gradOutput, scale)
    local scale = scale or 1
    local nframe = input:size(1)

    self.gradWeight:addmm(scale, gradOutput:t(), input)
    self.gradBias:addmv(scale, gradOutput:t(), input.new(nframe):fill(1))

    return self.gradWeight
end
