--One layer deconvnet with padding
require 'GaussianCriterion'
require 'cutorch'
require 'cunn'
require 'SpatialZeroPaddingCUDA'
cuda = true
continuous = true

---Required 
batchSize = 128 -- size of mini-batches
learningRate = 0.03 -- Learning rate used in AdaGrad

initrounds = 20 -- Amount of intialization rounds in AdaGrad

trsize = 50000-80 -- Size of training set
tesize = 10000-16 -- Size of test set

-- Loading data
-- trainData is table with field 'data' which contains the data
trainData, testData = loadCifar(trsize,tesize,false)

trainData.data = trainData.data:cuda()
testData.data = testData.data:cuda()



-- Model Specific parameters
filter_size = 5
stride = 2
dim_hidden = 100
input_size = 32 --NB this is done later (line 129)
pad1 = 2 --NB new size must be divisible with filtersize
pad2 = 2
colorchannels = 3
total_output_size = colorchannels * input_size ^ 2
feature_maps = 32

hidden_dec = 50

map_size = 16
factor = stride


encoder = nn.Sequential()
----------------------------   CUDA:    ----------------------------------------------------
encoder:add(nn.Transpose({1,4},{1,3},{1,2}))
encoder:add(nn.SpatialZeroPaddingCUDA(pad1,pad2,pad1,pad2))
encoder:add(nn.SpatialConvolutionCUDA(colorchannels,feature_maps,filter_size,filter_size))
encoder:add(nn.SpatialMaxPoolingCUDA(2,2,2,2))
encoder:add(nn.Transpose({4,1},{4,2},{4,3}))
-- ---------------------------         Regular:        --------------------------
--encoder:add(nn.SpatialZeroPadding(pad1,pad2,pad1,pad2))
--encoder:add(nn.SpatialConvolution(colorchannels,feature_maps,filter_size,filter_size))
--encoder:add(nn.SpatialMaxPooling(2,2,2,2))

encoder:add(nn.Threshold(0,1e-6))

encoder:add(nn.Reshape(feature_maps * map_size * map_size))

local z = nn.ConcatTable()
z:add(nn.LinearCR(feature_maps * map_size * map_size, dim_hidden))
z:add(nn.LinearCR(feature_maps * map_size * map_size, dim_hidden))

encoder:add(z)

local decoder = nn.Sequential()
decoder:add(nn.LinearCR(dim_hidden, feature_maps * map_size * map_size))
decoder:add(nn.Threshold(0,1e-6))

decoder:add(nn.Reshape(feature_maps,map_size,map_size))
decoder:add(nn.Transpose({2,3},{3,4}))

decoder:add(nn.Reshape(map_size*map_size*batchSize,feature_maps))
decoder:add(nn.LinearCR(feature_maps,hidden_dec))
decoder:add(nn.Threshold(0,1e-6))

decoder2 = nn.ConcatTable()
decoder2:add(nn.LinearCR(hidden_dec, colorchannels*factor*factor))
decoder2:add(nn.LinearCR(hidden_dec, colorchannels*factor*factor))

decoder3 = nn.ParallelTable()
decoder3:add(nn.Sigmoid())
decoder3:add(nn.Copy())

decoder4 = nn.ParallelTable()
decoder4:add(nn.Reshape(batchSize,total_output_size))
decoder4:add(nn.Reshape(batchSize,total_output_size))

decoder5 = nn.ParallelTable()
decoder5:add(nn.Copy('torch.CudaTensor','torch.DoubleTensor'))
decoder5:add(nn.Copy('torch.CudaTensor','torch.DoubleTensor'))


model = nn.Sequential()
model:add(encoder)
model:add(nn.Reparametrize(dim_hidden))
model:add(decoder)
model:add(decoder2)
model:add(decoder3)
model:add(decoder4)
model:add(decoder5)

encoder:cuda()
decoder:cuda()
decoder2:cuda()
decoder3:cuda()
decoder4:cuda()

