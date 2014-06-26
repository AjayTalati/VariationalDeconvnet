require 'cutorch'
require 'cunn'
require 'SpatialZeroPaddingCUDA'

cuda = true

batchSize = 128 -- size of mini-batches
learningRate = 0.005 -- Learning rate used in AdaGrad

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
stride = 1
dim_hidden = 25
input_size = 28
pad1 = 2 --NB new size must be divisible with filtersize
pad2 = 2
total_output_size = 1 * input_size ^ 2
feature_maps = 16
colorchannels = 1

map_size = 28^2
--factor = input_size/ 16

encoder = nn.Sequential()
encoder:add(nn.Transpose({1,4},{1,3},{1,2}))
encoder:add(nn.SpatialZeroPaddingCUDA(pad1,pad2,pad1,pad2))
encoder:add(nn.SpatialConvolutionCUDA(colorchannels,feature_maps,filter_size,filter_size,stride,stride))
encoder:add(nn.Transpose({4,1},{4,2},{4,3}))
encoder:add(nn.Threshold(0,1e-6))

encoder:add(nn.Reshape(feature_maps * map_size))

local z = nn.ConcatTable()
z:add(nn.LinearCR(feature_maps * map_size, dim_hidden))
z:add(nn.LinearCR(feature_maps * map_size, dim_hidden))

encoder:add(z)

local decoder = nn.Sequential()
decoder:add(nn.LinearCR(dim_hidden, feature_maps * map_size))
decoder:add(nn.Threshold(0,1e-6))

decoder:add(nn.Reshape(batchSize,feature_maps,input_size,input_size))
decoder:add(nn.SpatialZeroPadding(pad1,pad2,pad1,pad2))
decoder:add(nn.SpatialConvolution(feature_maps,feature_maps,filter_size,filter_size,stride,stride))

decoder:add(nn.Sum(2))
decoder:add(nn.Sigmoid())
decoder:add(nn.Reshape(batchSize,total_output_size))

model = nn.Sequential()
model:add(encoder)
model:add(nn.Reparametrize(dim_hidden))
model:add(decoder)
model:add(nn.Copy('torch.CudaTensor', 'torch.DoubleTensor'))

encoder:cuda()
decoder:cuda()
