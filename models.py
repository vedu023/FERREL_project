import torch
import torch.nn as nn 
import torchvision.models as m 


class conv_block(nn.Module):
    pass  

class Att_block():
    pass 

class head():
    pass 

class neck():
    pass
    
class MLP_block(nn.Module):
    pass
    
class model_1st_pass(nn.Module):
    
    def __init__(self):
        super(model_1st_pass, self).__init__()
        
        self.model1 = m.swin_v2_b(weights = m.Swin_V2_B_Weights.DEFAULT)
        
        
    def forward(self, x):
        return self.model1(x)
    
if __name__ == '__main__':
    
    model = model_1st_pass()
    print(model.model1.head.in_features)