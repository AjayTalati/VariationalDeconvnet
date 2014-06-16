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

foldername = 'results/MNIST/test_folder'
require foldername .. '/config'

 
------------------------------------------------------------------------------------------------------------

model = torch.load(fname .. '/model')
weights, gradients = model:getParameters()
weights:copy(torch.load(fname .. '/weights.t7'))
features_layer1 = weights[{{1,feature_maps*colorchannels*map_size*map_size}}]:reshape(feature_maps,colorchannels,filtersize,filtersize)

features_layer1 = {}
for i=1,feature_maps do
	table.insert(features_layer1,featuremaps[{{i},{1},{},{}}]:squeeze())
end
gfx.image(features_layer1,zoom=20)

trainData, testData = loadMnist(trsize,tesize)
data = testData.data[{{1,100},{},{},{}}]


function display(reconstruction, input)
  gfx.image({input:reshape(colorchannels,input_size,input_size), reconstruction:reshape(colorchannels,input_size,input_size)}, {zoom=9, legends={'Input', 'Reconstruction'}})
end

function plot_lowerbound() -- add testlowerbound later
	lowerbound = torch.load(foldername .. 'lowerbound.t7')
	lowerbound_test = torch.load(foldername .. 'lowerbound_test.t7')	
	gfx.chart(lowerbound,{chart='line'})
	gfx.chart(lowerbound_test, {chart='line'})
end