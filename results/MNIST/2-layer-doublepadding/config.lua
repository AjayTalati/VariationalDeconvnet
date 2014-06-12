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
filter_size_2 = 5
stride = 2
dim_hidden = 25
input_size = 28 --NB this is done later (line 129)
pad1 = 1 --NB new size must be divisible with filtersize
pad2 = 2
pad_2 = (filter_size_2-1)/2
total_output_size = 1 * input_size ^ 2
feature_maps = 15
feature_maps_2 = feature_maps*2

map_size = 14
map_size_2 = map_size
factor = input_size/map_size

--layer1
encoder = nn.Sequential()
encoder:add(nn.SpatialZeroPaddingC(pad1,pad2,pad1,pad2))
encoder:add(nn.SpatialConvolution(1,feature_maps,filter_size,filter_size,stride,stride))
encoder:add(nn.Threshold(0,0))
--layer2
encoder:add(nn.SpatialZeroPaddingC(pad_2,pad_2,pad_2,pad_2))
encoder:add(nn.SpatialConvolution(feature_maps,feature_maps_2,filter_size_2,filter_size_2,1,1))
encoder:add(nn.Threshold(0,0))
encoder:add(nn.Reshape(feature_maps_2 * map_size_2^2))

local z = nn.ConcatTable()
z:add(nn.LinearCR(feature_maps_2 * map_size_2^2, dim_hidden))
z:add(nn.LinearCR(feature_maps_2 * map_size_2^2, dim_hidden))

encoder:add(z)

local decoder = nn.Sequential()
decoder:add(nn.LinearCR(dim_hidden, feature_maps_2 * map_size_2^2))
decoder:add(nn.Threshold(0,0))
--layer2
decoder:add(nn.Reshape(batchSize,feature_maps_2,map_size,map_size))
decoder:add(nn.SpatialZeroPaddingC(pad_2,pad_2,pad_2,pad_2))
decoder:add(nn.SpatialConvolution(feature_maps_2,feature_maps,filter_size_2,filter_size_2,1,1))
--layer1
decoder:add(nn.Reshape((map_size^2)*batchSize,feature_maps))
decoder:add(nn.SpatialDeconvolution(feature_maps,1,factor))
decoder:add(nn.Sigmoid())
decoder:add(nn.Reshape(batchSize,total_output_size))

model = nn.Sequential()
model:add(encoder)
model:add(nn.Reparametrize(dim_hidden))
model:add(decoder)