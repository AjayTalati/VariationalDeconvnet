require 'sys'
require 'xlua'
require 'torch'
require 'nn'

require 'LinearCR'
require 'Reparametrize'
require 'SpatialDeconvolution'

require 'load'

gfx = require 'gfx.js'

-- Remember to start server!
-- luajit -lgfx.start
-- luajit -lgfx.stop

fname = 'results/CIFAR-10/1-layer-16'
require (fname .. '/config')

 
------------------------------------------------------------------------------------------------------------

--model = torch.load(fname .. '/model')

--trainData, testData = loadMnist(trsize,tesize)
--data = testData.data[{{1,100},{},{},{}}]
--weights, gradients = model:getParameters()

function display_reconstruction(input)	
	f = model:forward(input)
		for i = 1,10 do
		reconstruction = f[{{i},{}}]
		target = input[{{i},{},{},{}}]
		gfx.image({target:reshape(colorchannels,input_size,input_size), reconstruction:reshape(colorchannels,input_size,input_size)}, {zoom=9, legends={'Target', 'Reconstruction'}})
	end
end

function display_weights(model, layers)
	local features_layer1_table = {}
	weights:copy(torch.load(fname .. '/weights.t7'))
	local features_layer1 = weights[{{1,feature_maps*colorchannels*filter_size*filter_size}}]:reshape(feature_maps,colorchannels,filter_size,filter_size)
	for i=1,feature_maps do
		table.insert(features_layer1_table,features_layer1[{{i},{1},{},{}}]:squeeze())
	end
	gfx.image(features_layer1,{zoom=20})
end

function plot_lowerbound() -- add testlowerbound later
	cutoff = 0
	lowerbound = torch.load(fname .. '/lowerbound.t7')
	lowerbound_test = torch.load(fname .. '/lowerbound_test.t7')
	print(lowerbound_test)
	print(lowerbound)
	values_train = torch.Tensor(lowerbound:size(1) 		- cutoff, 2)
	values_test = torch.Tensor(lowerbound_test:size(1) 	- cutoff, 2)
	values_train[{{},{2}}] 	= lowerbound[{{1+cutoff		,lowerbound:size(1)}}]
	values_test[{{},{2}}]  	= lowerbound_test[{{1+cutoff,lowerbound_test:size(1)}}]
	values_train[{{},{1}}] 	  = torch.linspace(50000*(cutoff+1),50000*lowerbound:size(1),     lowerbound:size(1)-cutoff)
	values_test[{{},{1}}] = torch.linspace(50000*(cutoff+1),50000*(lowerbound_test:size(1)),lowerbound_test:size(1)-cutoff)

	--gfx.chart({ values_train, values_test },{chart = 'line'})
	--gfx.chart({ values },{chart = 'line'})

	data = {
	    {
	        key = 'Train',
	        color = '#00f',
	        values = values_train,
	    },
	    {
	        key = 'Test',
	        color = '#0f0',
	        values = values_test,
	    },
	}
	gfx.chart(data, {
	   chart = 'line', -- or: bar, stacked, multibar, scatter
	   width = 600,
	   height = 450,
	   yLabel = 'testtitle',
	})
	print(lowerbound[lowerbound:size()])
	print(lowerbound_test[lowerbound_test:size()])
	print(lowerbound:max())
	print(lowerbound_test:max())


end

function plot_relevant_dims(weights)
	--NB this only works for folder results/MNIST/1-layer, need to make generic for all that use SpatialDeconvolution
	max_filters_conv = feature_maps*colorchannels*filter_size*filter_size
	max_bias_conv = max_filters_conv + feature_maps

	max_mu_enc = max_bias_conv + dim_hidden * feature_maps * map_size
	max_bias_mu_enc  = max_mu_enc + dim_hidden

	max_sig_enc = max_bias_mu_enc + dim_hidden * feature_maps * map_size
	max_bias_sig_enc = max_sig_enc + dim_hidden

	max_fc_dec = max_bias_sig_enc + dim_hidden * feature_maps * map_size
	max_bias_fc_dec = max_fc_dec + map_size * feature_maps

	max_deconv = max_bias_fc_dec + colorchannels*filter_size*filter_size*feature_maps

	weights_conv = weights[{{1,max_filters_conv}}]
	weights_mu_enc = weights[{{max_bias_conv+1,max_mu_enc}}]
	weights_sig_enc = weights[{{max_bias_mu_enc+1,max_sig_enc}}]
	weights_fc_dec = weights[{{max_bias_sig_enc+1,max_fc_dec}}]
	weights_deconv = weights[{{max_bias_fc_dec+1,weights:size(1)}}]

	weights_mu_enc = weights_mu_enc:reshape(feature_maps, 14, 14, dim_hidden)
	weights_relevance = torch.Tensor(weights_mu_enc:size(4))

	for i = 1,weights_mu_enc:size(4) do
		weights_relevance[i] = torch.norm(weights_mu_enc[{{},{},{},{i}}])
	end
	gfx.chart(weights_relevance,{chart='line'})
end


--display_reconstruction(data)
--display_weights(model,1)
--plot_relevant_dims(weights)
plot_lowerbound()
