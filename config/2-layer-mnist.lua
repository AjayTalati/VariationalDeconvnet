batchSize = 100
learningRate = 0.05

-- Model Specific parameters
filter_size = 5
filter_size_2 = 5
stride = 2
stride_2 = 2
dim_hidden = 25
input_size = 28 --NB this is done later (line 129)
pad_11 = 1
pad_12 = 2
pad_21 = 1
pad_22 = 2
total_output_size = 1 * input_size ^ 2
feature_maps = 15
feature_maps_2 = feature_maps*2

map_size = 14
map_size_2 = 7
factor = input_size/map_size
factor_2 = map_size/map_size_2

--layer1
encoder = nn.Sequential()
encoder:add(nn.SpatialZeroPaddingC(pad_11,pad_12,pad_11,pad_12))
encoder:add(nn.SpatialConvolution(1,feature_maps,filter_size,filter_size,stride,stride))
encoder:add(nn.Threshold(0,0))
--layer2
encoder:add(nn.SpatialZeroPaddingC(pad_21,pad_22,pad_21,pad_22))
encoder:add(nn.SpatialConvolution(feature_maps,feature_maps_2,filter_size_2,filter_size_2,stride_2,stride_2))
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
decoder:add(nn.Reshape((map_size_2^2)*batchSize,feature_maps_2))
decoder:add(nn.SpatialDeconvolution(feature_maps_2,feature_maps,factor_2))
decoder:add(nn.Threshold(0,0))
--layer1
decoder:add(nn.Reshape((map_size^2)*batchSize,feature_maps))
decoder:add(nn.SpatialDeconvolution(feature_maps,1,factor))
decoder:add(nn.Sigmoid())
decoder:add(nn.Reshape(batchSize,total_output_size))

model = nn.Sequential()
model:add(encoder)
model:add(nn.Reparametrize(dim_hidden))
model:add(decoder)