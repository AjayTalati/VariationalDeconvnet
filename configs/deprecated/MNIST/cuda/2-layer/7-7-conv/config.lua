require 'cutorch'
require 'cunn'
require 'SpatialZeroPaddingCUDA'

cuda = true

batchSize = 128 -- size of mini-batches
learningRate = 0.05 -- Learning rate used in AdaGrad

initrounds = 5 -- Amount of intialization rounds in AdaGrad

trsize = 50000-80 -- Size of training set
tesize = 10000-16 -- Size of test set

-- Loading data
-- trainData is table with field 'data' which contains the data
trainData, testData = loadMnist(trsize,tesize)

trainData.data = trainData.data:cuda()
testData.data = testData.data:cuda()

-- Model Specific parameters
filter_size = 5
filter_size_2 = 5
stride = 2 
dim_hidden = 25
input_size = 28 --NB this is done later (line 129)
pad1 = 2 --NB new size must be divisible with filtersize
pad2 = 2
pad_2 = (filter_size_2-1)/2
total_output_size = 1 * input_size ^ 2
feature_maps = 16
feature_maps_2 = feature_maps*2

factor = 4

map_size = 7
map_size_2 = 7

--hidden_dec should be in order of: featuremaps * filtersize^2 / (16+factor^2)
hidden_dec = 50


colorchannels = 1
encoder = nn.Sequential()

--layer1
encoder:add(nn.SpatialZeroPadding(pad1,pad2,pad1,pad2))
encoder:add(nn.SpatialConvolution(colorchannels,feature_maps,filter_size,filter_size))
encoder:add(nn.SpatialMaxPooling(4,4,4,4))
encoder:add(nn.Threshold(0,1e-6))

--layer2
encoder:add(nn.SpatialZeroPadding(pad_2,pad_2,pad_2,pad_2)) 
encoder:add(nn.SpatialConvolution(feature_maps,feature_maps_2,filter_size_2,filter_size_2))
encoder:add(nn.Threshold(0,1e-6))

encoder:add(nn.Reshape(feature_maps_2 * map_size_2^2))

local z = nn.ConcatTable()
z:add(nn.LinearCR(feature_maps_2 * map_size_2^2, dim_hidden))
z:add(nn.LinearCR(feature_maps_2 * map_size_2^2, dim_hidden))

encoder:add(z)

local decoder = nn.Sequential()
decoder:add(nn.LinearCR(dim_hidden, feature_maps_2 * map_size_2^2))
decoder:add(nn.Threshold(0,1e-6))
--layer2
decoder:add(nn.Reshape(batchSize,feature_maps_2,map_size_2,map_size_2))
decoder:add(nn.SpatialZeroPadding(pad_2,pad_2,pad_2,pad_2))
decoder:add(nn.SpatialConvolution(feature_maps_2,feature_maps,filter_size_2,filter_size_2))
--layer1
decoder:add(nn.Reshape(feature_maps,map_size,map_size))
decoder:add(nn.Transpose({2,3},{3,4}))
decoder:add(nn.Reshape(map_size*map_size*batchSize,feature_maps))
decoder:add(nn.LinearCR(feature_maps,hidden_dec))
decoder:add(nn.Threshold(0,1e-6))
decoder:add(nn.LinearCR(hidden_dec,colorchannels*factor*factor)) 
decoder:add(nn.Sigmoid())
decoder:add(nn.Reshape(batchSize,total_output_size))

model = nn.Sequential()
model:add(encoder)
model:add(nn.Reparametrize(dim_hidden))
model:add(decoder)
model:add(nn.Copy('torch.CudaTensor','torch.DoubleTensor'))

encoder:cuda()
decoder:cuda()
