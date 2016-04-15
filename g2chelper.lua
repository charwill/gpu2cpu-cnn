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


function checkpoint.restoreModules(oldNet)
   local newNet= oldNet:clone()
   for i = 1, #oldNet do  
      local oldNetLayer= oldNet:get(i)
      local layer_type= torch.type(oldNetLayer)     
      local module_type= tostring(string.split(layer_type,'%.')[1])
      local module_name= tostring(string.split(layer_type,'%.')[2])
    
      if module_name=='SpatialConvolution' or 'SpatialBatchNormalization' then
         if newNet.modules[i].gradInput==nil  then table.insert(newNet.modules[i],'gradInput'); newNet.modules[i].gradInput =torch.FloatTensor(); newNet.modules[i][1]=nil end
         if newNet.modules[i].finput==nil     then table.insert(newNet.modules[i],'finput');  newNet.modules[i].finput =torch.FloatTensor(); newNet.modules[i][1]=nil end
         if newNet.modules[i].fgradInput==nil then table.insert(newNet.modules[i],'fgradInput'); newNet.modules[i].fgradInput =torch.FloatTensor(); newNet.modules[i][1]=nil end

         if newNet.modules[i].gradWeight==nil and newNet.modules[i].weight then table.insert(newNet.modules[i],'gradWeight'); newNet.modules[i].gradWeight =torch.FloatTensor(); newNet.modules[i][1]=nil end
         if newNet.modules[i].gradBias==nil and newNet.modules[i].bias     then table.insert(newNet.modules[i],'gradBias'); newNet.modules[i].gradBias =torch.FloatTensor(); newNet.modules[i][1]=nil end
         if newNet.modules[i].gradBias and newNet.modules[i].bias          then newNet.modules[i].gradBias= newNet.modules[i].bias:clone():zero() end
         if newNet.modules[i].gradWeight and newNet.modules[i].weight      then newNet.modules[i].gradWeight= newNet.modules[i].weight:clone():zero() end 
     end 
      if newNet.modules[i].train==nil then table.insert(newNet.modules[i], 'train'); newNet.modules[i].train= true; newNet.modules[i][1]=nil end 

   end --for  
   return newNet   
end

return checkpoint
