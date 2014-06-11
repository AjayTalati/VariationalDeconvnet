-- General parameters
batchSize = 100
learningRate = 0.05

-- Model Specific parameters
filter_size = 5
stride = 2
dim_hidden = 25
input_size = 28 --NB this is done later (line 129)
pad1 = 1 --NB new size must be divisible with filtersize
pad2 = 1
total_output_size = input_size ^ 2
feature_maps = 10
input_planes = 1

map_size = 15 ^2
factor = (input_size+pad1+pad2)/ ((input_size+pad1+pad2)/stride)

encoder = nn.Sequential()
encoder:add(nn.SpatialZeroPaddingC(pad1,pad2,pad1,pad2))
encoder:add(nn.SpatialConvolution(input_planes,feature_maps,filter_size,filter_size,stride,stride))
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
decoder:add(nn.SpatialDeconvolution(feature_maps,input_planes,factor))
decoder:add(nn.Sigmoid())
decoder:add(nn.Reshape(batchSize,total_output_size))

model = nn.Sequential()
model:add(encoder)
model:add(nn.Reparametrize(dim_hidden))
model:add(decoder)