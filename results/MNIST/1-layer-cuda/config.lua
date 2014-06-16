--One layer deconvnet with padding
require 'cutorch'
require 'cunn'

---Required 
batchSize = 100 -- size of mini-batches
learningRate = 0.05 -- Learning rate used in AdaGrad

initrounds = 5 -- Amount of intialization rounds in AdaGrad

trsize = 50000 -- Size of training set
tesize = 10000 -- Size of test set

-- Loading data
-- trainData is table with field 'data' which contains the data
trainData, testData = loadMnist(trsize,tesize)

-- Model Specific parameters
filter_size = 5
stride = 2
dim_hidden = 25
input_size = 28 --NB this is done later (line 129)
pad1 = 1 --NB new size must be divisible with filtersize
pad2 = 2
colorchannels = 1
total_output_size = colorchannels * input_size ^ 2
feature_maps = 15

map_size = 14^2
factor = input_size/14


encoder = nn.Sequential()
encoder:add(nn.SpatialZeroPadding(pad1,pad2,pad1,pad2))
encoder:add(nn.SpatialConvolution(colorchannels,feature_maps,filter_size,filter_size,stride,stride))
encoder:add(nn.Threshold(0,0))

encoder:add(nn.Reshape(feature_maps * map_size))

local z = nn.ConcatTable()
z:add(nn.LinearCR(feature_maps * map_size, dim_hidden))
z:add(nn.LinearCR(feature_maps * map_size, dim_hidden))

local decoder = nn.Sequential()
decoder:add(nn.LinearCR(dim_hidden, feature_maps * map_size))
decoder:add(nn.Threshold(0,0))



decoder:add(nn.Reshape(map_size*batchSize,feature_maps))
decoder:add(nn.SpatialDeconvolution(feature_maps,colorchannels,factor))
decoder:add(nn.Sigmoid())
decoder:add(nn.Reshape(batchSize,total_output_size))

model = nn.Sequential()
model:add(encoder)
model:add(nn.Copy('torch.CudaTensor', 'torch.DoubleTensor'))
model:add(z)
model:add(nn.Reparametrize(dim_hidden))
model:add(nn.Copy('torch.DoubleTensor', 'torch.CudaTensor'))
model:add(decoder)

encoder:cuda()
decoder:cuda()
