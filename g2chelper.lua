--
--  Copyright (c) 2016, TCL research Hong Kong, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--


local checkpoint = {}
function checkpoint.replaceModules(oldNet)
   local newNet= oldNet:clone():float()
   for i = 1, #oldNet do  
      local oldNetLayer= oldNet:get(i)
      local layer_type= torch.type(oldNetLayer)     
      local module_type= tostring(string.split(layer_type,'%.')[1])
      local module_name= tostring(string.split(layer_type,'%.')[2])
      local is_conv=(layer_type == 'cudnn.SpatialConvolution' or layer_type == 'nn.SpatialConvolution')
      local is_relu= (layer_type == 'cudnn.ReLU' or layer_type == 'nn.ReLU')           
      if module_type=='cudnn' then
         if is_conv then
            local nInputPlane, nOutputPlane = oldNetLayer.nInputPlane, oldNetLayer.nOutputPlane
            local kW, kH, dW, dH = oldNetLayer.kW, oldNetLayer.kH, oldNetLayer.dW, oldNetLayer.dH
            local padW= oldNetLayer.padW or 0
            local padH= oldNetLayer.padH or padW
            newNet.modules[i]=nn.SpatialConvolution(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)          
            newNet.modules[i].weight:copy(oldNetLayer.weight);newNet.modules[i].bias:copy(oldNetLayer.bias)
         end
         
         if is_relu then
            newNet.modules[i]=nn.ReLU(true)
         end
      end 
      newNet.modules[i].gradInput=nil
      newNet.modules[i].finput=nil
      newNet.modules[i].fgradInput=nil
      newNet.modules[i].gradBias=nil
      newNet.modules[i].gradWeight=nil
   end --for  
   return newNet:float()   
end




return checkpoint
