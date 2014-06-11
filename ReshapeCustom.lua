local ReshapePad, parent = torch.class('nn.ReshapePad', 'nn.Module')

function ReshapePad:__init(pad,...)
   parent.__init(self)
   local arg = {...}
   self.pad = pad

   self.size = torch.LongStorage()
   self.batchsize = torch.LongStorage()
   local n = #arg
   if n == 1 and torch.typename(arg[1]) == 'torch.LongStorage' then
      self.size:resize(#arg[1]):copy(arg[1])
   else
      self.size:resize(n)
      for i=1,n do
         self.size[i] = arg[i]
      end
   end
   self.nelement = 1
   self.batchsize:resize(#self.size+1)
   for i=1,#self.size do
      self.nelement = self.nelement * self.size[i]
      self.batchsize[i+1] = self.size[i]
   end
end

function ReshapePad:updateOutput(input)
   input = input:contiguous()
   local nelement = input:nElement()
   if nelement == self.nelement and input:size(1) ~= 1 then
      input:resize(self.size)
      self.output = torch.zeros(self.size[1],self.size[2],self.size[3]+self.pad,self.size[4]+self.pad)
      self.output[{{},{},{self.pad,self.size[3]-self.pad},{self.pad,self.size[4]-self.pad}}] = input
   else
      self.batchsize[1] = input:size(1)

      input:resize(self.batchsize)

      self.output = torch.zeros(self.batchsize[1],self.batchsize[2],self.batchsize[3]+self.pad,self.batchsize[4]+self.pad)
      self.output[{{},{},{self.pad,self.batchsize[3]-self.pad},{self.pad,self.batchsize[4]-self.pad}}] = input
   end
   return self.output
end

function ReshapePad:updateGradInput(input, gradOutput)
   gradOutput = gradOutput:contiguous()
   self.gradInput:set(gradOutput):resizeAs(input)
   return self.gradInput
end
