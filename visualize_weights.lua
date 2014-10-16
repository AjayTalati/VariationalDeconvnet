require 'image'

param = torch.load('blockparam2.t7')

img = torch.Tensor(16,5,5)

for i = 1,16 do
	block = torch.reshape(param[9][{{},{i}}],5,5)
	block:add(-torch.min(block))
	block:div(torch.max(block))
	img[{{i},{},{}}] = block
end

img = torch.reshape(img,80,5)

image.savePNG('png/img.png',img)
