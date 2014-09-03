--Two layer deconvnet

-- Model Specific parameters
filter_size = 5
filter_size_2 = 5

stride = 2
stride_2 = 2
dim_hidden = 25

input_size = 32
pad1 = 2 --NB new size must be divisible with filtersize
pad2 = 2
pad2_1 = 2
pad2_2 = 2
total_output_size = 1 * input_size ^ 2
feature_maps = 15
feature_maps_2 = feature_maps*2

map_size = 16
map_size_2 = 8

factor = 2
factor_2 = 2

colorchannels = 1

--layer1
encoder = nn.Sequential()
encoder:add(nn.SpatialZeroPadding(pad1,pad2,pad1,pad2))
encoder:add(nn.SpatialConvolutionMM(colorchannels,feature_maps,filter_size,filter_size))
encoder:add(nn.SpatialMaxPooling(2,2,2,2))
encoder:add(nn.Threshold(0,1e-6))

--layer2
encoder:add(nn.SpatialZeroPadding(pad2_1,pad2_2,pad2_1,pad2_2))
encoder:add(nn.SpatialConvolutionMM(feature_maps,feature_maps_2,filter_size_2,filter_size_2))
encoder:add(nn.SpatialMaxPooling(2,2,2,2))
encoder:add(nn.Threshold(0,1e-6))
encoder:add(nn.Reshape(feature_maps_2 * map_size_2 * map_size_2))

local z = nn.ConcatTable()
z:add(nn.LinearCR(feature_maps_2 * map_size_2^2, dim_hidden))
z:add(nn.LinearCR(feature_maps_2 * map_size_2^2, dim_hidden))

encoder:add(z)

local decoder = nn.Sequential()
decoder:add(nn.LinearCR(dim_hidden, feature_maps_2 * map_size_2 * map_size_2))
decoder:add(nn.Threshold(0,1e-6))

decoder:add(nn.Reshape(batchSize, feature_maps_2, map_size_2, map_size_2))
decoder:add(nn.Transpose({2,3},{3,4}))

--layer2
decoder:add(nn.Reshape(batchSize,feature_maps_2,map_size_2,map_size_2))
decoder:add(nn.Transpose({2,3},{3,4}))
decoder:add(nn.Reshape(map_size_2*map_size_2*batchSize,feature_maps_2))
decoder:add(nn.LinearCR(feature_maps_2,hidden_dec_2))
decoder:add(nn.Threshold(0,1e-6))
decoder:add(nn.LinearCR(hidden_dec_2,feature_maps*factor*factor))
decoder:add(nn.Threshold(0,1e-6))
--layer1
decoder:add(nn.LinearCR(feature_maps*factor*factor,hidden_dec))
decoder:add(nn.Threshold(0,1e-6))
decoder:add(nn.LinearCR(hidden_dec,colorchannels*factor^4)) 
decoder:add(nn.Sigmoid())
decoder:add(nn.Reshape(batchSize,total_output_size))

decoder:add(nn.Reshape(batchSize, total_output_size))

model = nn.Sequential()
model:add(encoder)
model:add(nn.Reparametrize(dim_hidden))
model:add(decoder)