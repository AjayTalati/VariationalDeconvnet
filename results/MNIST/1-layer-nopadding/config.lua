batchSize = 100
learningRate = 0.05

initrounds = 5

trsize = 50000
tesize = 10000

trainData, testData = loadCifar(trsize,tesize,false)

-- Model Specific parameters
filter_size = 4
stride = 4
dim_hidden = 25
input_size = 32 

total_output_size = 3 * input_size ^ 2
feature_maps = 10

map_size = (input_size / stride) ^ 2

-- map_size = 16 ^ 2

encoder = nn.Sequential()
encoder:add(nn.SpatialConvolution(3,feature_maps,filter_size,filter_size,stride,stride))
encoder:add(nn.Threshold(0,0))
encoder:add(nn.Reshape(feature_maps * map_size))

local z = nn.ConcatTable()
z:add(nn.LinearCR(feature_maps * map_size, dim_hidden))
z:add(nn.LinearCR(feature_maps * map_size, dim_hidden))

encoder:add(z)

local decoder = nn.Sequential()
decoder:add(nn.LinearCR(dim_hidden, feature_maps * map_size))
decoder:add(nn.Threshold(0,0))
--Possibly move to (batchsize,feature_maps * mapsize): saves TWO reshapes
decoder:add(nn.Reshape(map_size*batchSize,feature_maps))
decoder:add(nn.SpatialDeconvolution(feature_maps,3,stride))
decoder:add(nn.Sigmoid())
decoder:add(nn.Reshape(batchSize,total_output_size))

model = nn.Sequential()
model:add(encoder)
model:add(nn.Reparametrize(dim_hidden))
model:add(decoder)