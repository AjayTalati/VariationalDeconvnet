-- General parameters
batchSize = 100
learningRate = 0.05

-- Model Specific parameters
filter_size = 5
stride = 2
dim_hidden = 25
input_size = 28 --NB this is done later (line 129)
pad1 = 1 --NB new size must be divisible with filtersize
pad2 = 2
total_output_size = 1 * input_size ^ 2
feature_maps = 20

map_size = 14 ^2
factor = input_size/14

encoder = nn.Sequential()
encoder:add(nn.SpatialZeroPaddingC(pad1,pad2,pad1,pad2))
encoder:add(nn.SpatialConvolution(1,feature_maps,filter_size,filter_size,stride,stride))
encoder:add(nn.Threshold(0,0))
encoder:add(nn.Reshape(feature_maps * map_size))

local z = nn.ConcatTable()
z:add(nn.LinearCR(feature_maps * map_size, dim_hidden))
z:add(nn.LinearCR(feature_maps * map_size, dim_hidden))

encoder:add(z)

local decoder = nn.Sequential()
decoder:add(nn.LinearCR(dim_hidden, feature_maps * map_size))
decoder:add(nn.Threshold(0,0))
decoder:add(nn.Reshape(map_size*batchSize,feature_maps))
decoder:add(nn.SpatialDeconvolution(feature_maps,1,factor))
decoder:add(nn.Sigmoid())
decoder:add(nn.Reshape(batchSize,total_output_size))

model = nn.Sequential()
model:add(encoder)
model:add(nn.Reparametrize(dim_hidden))
model:add(decoder)