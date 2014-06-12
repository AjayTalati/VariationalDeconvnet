---TEMPLATE CONFIG FILE

---Required 
batchSize = 100 -- size of mini-batches
learningRate = 0.05 -- Learning rate used in AdaGrad

initrounds = 5 -- Amount of intialization rounds in AdaGrad

trsize = 50000 -- Size of training set
tesize = 10000 -- Size of test set

-- Loading data
-- trainData is table with field 'data' which contains the data
trainData, testData = loadCifar(trsize,tesize,false)

-- Model Specific parameters
local filter_size = 4

-- Model has to define encoder (used for KLD backpropagation) and model (the actual model to train)
encoder = nn.Sequential()

local z = nn.ConcatTable()
--NOTE LinearCR has TWO inputs that need to be added here
z:add(nn.LinearCR())
z:add(nn.LinearCR())

encoder:add(z)

local decoder = nn.Sequential()

model = nn.Sequential()
model:add(encoder)
model:add(nn.Reparametrize(dim_hidden))
model:add(decoder)