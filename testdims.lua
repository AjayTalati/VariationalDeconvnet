require 'torch'
require 'image'
gfx = require 'gfx.js'

set = torch.load('cifar-10-batches-t7/data_batch_1.t7', 'ascii')
images = set.data:t()

data = images[{{1,2},{}}]

image_test = images[{{1},{}}]
image_test = image_test:reshape(3,32,32)
gfx.image(image_test)
