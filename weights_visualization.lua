require 'image'

gfx = require 'gfx.js'

inputsize = 32
colorchannels = 3
featuremaps = 10
filtersize = 4
dimz = 25

featuremapsize = inputsize/filtersize

weights_all = torch.load('params/100_weight.t7')

-----------------------------------------------------------------------------------------
max_filters_conv = featuremaps*colorchannels*filtersize*filtersize
max_bias_conv = max_filters_conv + featuremaps


max_mu_enc = max_bias_conv + dimz * featuremaps * featuremapsize * featuremapsize
max_bias_mu_enc  = max_mu_enc + dimz

max_sig_enc = max_bias_mu_enc + dimz * featuremaps * featuremapsize * featuremapsize
max_bias_sig_enc = max_sig_enc + dimz

max_fc_dec = max_bias_sig_enc + dimz * featuremaps * featuremapsize * featuremapsize
max_bias_fc_dec = max_fc_dec + featuremapsize * featuremapsize * featuremaps

max_deconv = max_bias_fc_dec + colorchannels*filtersize*filtersize*featuremaps

-------------------------------------------------------------------------------------------

weights_conv = weights_all[{{1,max_filters_conv}}]
weights_mu_enc = weights_all[{{max_bias_conv+1,max_mu_enc}}]
weights_sig_enc = weights_all[{{max_bias_mu_enc+1,max_sig_enc}}]
weights_fc_dec = weights_all[{{max_bias_sig_enc+1,max_fc_dec}}]
weights_deconv = weights_all[{{max_bias_fc_dec+1,weights_all:size(1)}}]

convolutions = weights_conv:reshape(featuremaps,colorchannels,filtersize,filtersize)
gfx.image(convolutions[{{1},{3}}],{zoom=40})