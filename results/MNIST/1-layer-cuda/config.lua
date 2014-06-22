--One layer deconvnet with padding
require 'cutorch'
require 'cunn'
require 'SpatialZeroPaddingCUDA'

cuda = true

---Required 
batchSize = 128 -- size of mini-batches
learningRate = 0.03 -- Learning rate used in AdaGrad

initrounds = 10 -- Amount of intialization rounds in AdaGrad

trsize = 50000-80 -- Size of training set
tesize = 10000-16 -- Size of test set

-- Loading data
-- trainData is table with field 'data' which contains the data
trainData, testData = loadMnist(trsize,tesize)

trainData.data = trainData.data:cuda()
testData.data = testData.data:cuda()

-- Model Specific parameters
filter_size = 5
stride = 2
dim_hidden = 25
input_size = 28 --NB this is done later (line 129)
pad1 = 1 --NB new size must be divisible with filtersize
pad2 = 2
colorchannels = 1
total_output_size = colorchannels * input_size ^ 2
feature_maps = 16

map_size = 14^2
factor = input_size/14


encoder = nn.Sequential()
encoder:add(nn.Transpose({1,4},{1,3},{1,2}))
encoder:add(nn.SpatialZeroPaddingCUDA(pad1,pad2,pad1,pad2))
encoder:add(nn.SpatialConvolutionCUDA(colorchannels,feature_maps,filter_size,filter_size,stride,stride))
encoder:add(nn.Transpose({4,1},{4,2},{4,3}))
encoder:add(nn.Threshold(0,0))

encoder:add(nn.Reshape(feature_maps * map_size))

local z = nn.ConcatTable()
z:add(nn.LinearCR(feature_maps * map_size, dim_hidden))
z:add(nn.LinearCR(feature_maps * map_size, dim_hidden))

encoder:add(z)

local decoder = nn.Sequential()
decoder:add(nn.LinearCR(dim_hidden, feature_maps * map_size))
decoder:add(nn.Threshold(0,0))

decoder:add(nn.Reshape(feature_maps,14,14))
decoder:add(nn.Transpose({2,3},{3,4}))

decoder:add(nn.Reshape(map_size*batchSize,feature_maps))
-- decoder:add(nn.SpatialDeconvolution(feature_maps,colorchannels,factor))
decoder:add(nn.LinearCR(feature_maps,colorchannels*factor*factor))
decoder:add(nn.Sigmoid())
decoder:add(nn.Reshape(batchSize,total_output_size))

model = nn.Sequential()
model:add(encoder)
model:add(nn.Reparametrize(dim_hidden))
model:add(decoder)
model:add(nn.Copy('torch.CudaTensor', 'torch.DoubleTensor'))

encoder:cuda()
decoder:cuda()
