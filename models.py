import torch
import torch.nn as nn 
import torchvision.models as m 


class conv_block(nn.Module):
    
    def __init__(self, in_ch, out_ch):
        super(self, conv_block).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch), 
            nn.ReLU6(inplace=True),
            nn.BatchNorm2d(out_ch),
            nn.Conv2d(out_ch, out_ch),
            nn.ReLU6(inplace=True)
            )
        
    def forward(self, x):
        return self.conv(x)
    
    

class Att_block():
    def __init__(self, Q, V, K):
        super(Att_block, self).__init__()
        
        pass
    
    


class head():
    
    def __init__(self, no_class):
        super(self, head).__init__()
        
        self.head1 = MLP_block()
        self.head2 = MLP_block()
        
        
    def forward(self, x):
        h1 = self.head1(x)
        h2 = self.head2(x)
        
        return h1, h2
        

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