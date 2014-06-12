require 'sys'
require 'xlua'
require 'torch'
require 'nn'

require 'LinearCR'
require 'Reparametrize'
require 'SpatialDeconvolution'
require 'SpatialZeroPaddingC'
require 'load'

gfx = require 'gfx.js'

------------------------------------------------------------------------------------------------------------

require 'params/config'
fname_weights = 'params/100_weights.t7'
 
------------------------------------------------------------------------------------------------------------

model = torch.load('params/model')
weights, gradients = model:getParameters()
weights:copy(torch.load(fname_weights))
features_layer1 = weights[{{1,feature_maps*colorchannels*map_size*map_size}}]:reshape(feature_maps,colorchannels,filtersize,filtersize)

features_layer1 = {}
for i=1,feature_maps do
	table.insert(features_layer1,featuremaps[{{i},{1},{},{}}]:squeeze())
end
gfx.image(features_layer1,zoom=20)

trainData, testData = loadMnist(trsize,tesize)
print(testData.data:size())
data = testData.data[{{1,100},{},{},{}}]


function display(reconstruction, input)
  gfx.image({input:reshape(colorchannels,28,28), reconstruction:reshape(1,28,28)}, {zoom=9, legends={'Input', 'Reconstruction'}})
end