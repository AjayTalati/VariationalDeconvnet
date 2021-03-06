--One layer deconvnet with padding

-- Model Specific parameters
filter_size = 5
dim_hidden = 30
input_size = 32 
pad1 = 2 
pad2 = 2
colorchannels = 1
total_output_size = colorchannels * input_size ^ 2
feature_maps = 16

hidden_dec = 25

map_size = 16
factor = 2


encoder = nn.Sequential()
encoder:add(nn.SpatialZeroPadding(pad1,pad2,pad1,pad2))
encoder:add(nn.SpatialConvolutionMM(colorchannels,feature_maps,filter_size,filter_size))
encoder:add(nn.SpatialMaxPooling(2,2,2,2))
encoder:add(nn.Threshold(0,1e-6))

encoder:add(nn.Reshape(feature_maps * map_size * map_size))

local z = nn.ConcatTable()
z:add(nn.LinearCR(feature_maps * map_size * map_size, dim_hidden))
z:add(nn.LinearCR(feature_maps * map_size * map_size, dim_hidden))

encoder:add(z)

local decoder = nn.Sequential()
decoder:add(nn.LinearCR(dim_hidden, feature_maps * map_size * map_size))
decoder:add(nn.Threshold(0,1e-6))

--Reshape and transpose in order to upscale
decoder:add(nn.Reshape(batchSize, feature_maps, map_size, map_size))
decoder:add(nn.Transpose({2,3},{3,4}))

--Reshape and compute upscale with hidden dimensions
decoder:add(nn.Reshape(map_size * map_size * batchSize, feature_maps))
decoder:add(nn.LinearCR(feature_maps,hidden_dec))
decoder:add(nn.Threshold(0,1e-6))


decoder:add(nn.LinearCR(hidden_dec,colorchannels*factor*factor))
decoder:add(nn.Sigmoid())


decoder:add(nn.Reshape(batchSize,total_output_size))

model = nn.Sequential()
model:add(encoder)
model:add(nn.Reparametrize(dim_hidden))
model:add(decoder)