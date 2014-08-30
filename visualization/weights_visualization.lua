require 'image'

gfx = require 'gfx.js'
--th -lgfx.go

inputsize = 28
colorchannels = 1
featuremaps = 15
filtersize = 4
dim_hidden = 25
stride = 2
featuremapsize = 14

weights_all = torch.load('params/100_weights.t7')

-----------------------------------------------------------------------------------------
max_filters_conv = featuremaps*colorchannels*filtersize*filtersize
max_bias_conv = max_filters_conv + featuremaps

max_mu_enc = max_bias_conv + dim_hidden * featuremaps * map_size
max_bias_mu_enc  = max_mu_enc + dim_hidden

max_sig_enc = max_bias_mu_enc + dim_hidden * featuremaps * map_size
max_bias_sig_enc = max_sig_enc + dim_hidden

max_fc_dec = max_bias_sig_enc + dim_hidden * featuremaps * map_size
max_bias_fc_dec = max_fc_dec + map_size * featuremaps

max_deconv = max_bias_fc_dec + colorchannels*filtersize*filtersize*featuremaps

-------------------------------------------------------------------------------------------

weights_conv = weights_all[{{1,max_filters_conv}}]
weights_mu_enc = weights_all[{{max_bias_conv+1,max_mu_enc}}]
weights_sig_enc = weights_all[{{max_bias_mu_enc+1,max_sig_enc}}]
weights_fc_dec = weights_all[{{max_bias_sig_enc+1,max_fc_dec}}]
weights_deconv = weights_all[{{max_bias_fc_dec+1,weights_all:size(1)}}]

convolutions = weights_conv:reshape(featuremaps,colorchannels,filtersize,filtersize)
features = {}
for i=1,featuremaps do
	table.insert(features,convolutions[{{i},{1},{},{}}]:squeeze())
	gfx.image(features,{zoom=20, legends = {'','','','','','','','','','','','','','',''}})
end