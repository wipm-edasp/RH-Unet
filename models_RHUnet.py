import torch
import torch.nn as nn

class convBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(convBlock, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3,padding=1,bias=False),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3,padding=1,bias=False),
            nn.ReLU())
    def forward(self, x):
        x = self.cnn(x)
        return x

class upSampling(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(upSampling, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, middle_channels, kernel_size=3, padding=1,bias=False),
            nn.ReLU(),
            nn.Conv1d(middle_channels, middle_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.ConvTranspose1d(middle_channels, out_channels, kernel_size=2, stride=2, bias=False))
    def forward(self, x):
        x = self.cnn(x)
        return x        

class uNet8(nn.Module):
    def __init__(self, num_classes = 1 ):
        super(uNet8, self).__init__()
        self.enCode1 = convBlock(in_channels=1, out_channels=64)
        self.enCode2 = convBlock(in_channels=64, out_channels=128)
        self.enCode3 = convBlock(in_channels=128, out_channels=256)
        self.enCode4 = convBlock(in_channels=256, out_channels=256)
        self.enCode5 = convBlock(in_channels=256, out_channels=256)
        self.enCode6 = convBlock(in_channels=256, out_channels=256)
        self.enCode7 = convBlock(in_channels=256, out_channels=512)
        self.enCode8 = convBlock(in_channels=512, out_channels=512)
        self.Maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.deCode1 = upSampling(in_channels=512, middle_channels=256, out_channels=512)
        self.deCode2 = upSampling(in_channels=512*2, middle_channels=256, out_channels=512*1)
        self.deCode3 = upSampling(in_channels=512*2, middle_channels=256, out_channels=256)
        self.deCode4 = upSampling(in_channels=512*1, middle_channels=256, out_channels=256)
        self.deCode5 = upSampling(in_channels=512, middle_channels=256, out_channels=256)
        self.deCode6 = upSampling(in_channels=256*2, middle_channels=128, out_channels=256)
        self.deCode7 = upSampling(in_channels=256*2, middle_channels=128, out_channels=128)   
        self.deCode8 = upSampling(in_channels=128*2, middle_channels=128, out_channels=64)
        self.lastLayer = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv1d(64, num_classes, kernel_size=1, bias=False))  
        
    def forward(self, x):
        enc1 = self.enCode1(x)
        enc1_pool = self.Maxpool(enc1)
        enc2 = self.enCode2(enc1_pool)
        enc2_pool = self.Maxpool(enc2)
        enc3 = self.enCode3(enc2_pool)
        enc3_pool = self.Maxpool(enc3)
        enc4 = self.enCode4(enc3_pool)
        enc4_pool = self.Maxpool(enc4)
        enc5 = self.enCode5(enc4_pool) 
        enc5_pool = self.Maxpool(enc5) 
        enc6 = self.enCode6(enc5_pool)
        enc6_pool = self.Maxpool(enc6)        
        enc7 = self.enCode7(enc6_pool)
        enc7_pool = self.Maxpool(enc7)
        enc8 = self.enCode8(enc7_pool)
        enc8_pool = self.Maxpool(enc8)                       
        
        dec1 = self.deCode1(enc8_pool)   
        dec2 = self.deCode2(torch.cat((dec1, enc8), dim=1))   
        dec3 = self.deCode3(torch.cat((dec2, enc7), dim=1))
        dec4 = self.deCode4(torch.cat((dec3, enc6), dim=1))
        dec5 = self.deCode5(torch.cat((dec4, enc5), dim=1))        
        dec6 = self.deCode6(torch.cat((dec5, enc4), dim=1))        
        dec7 = self.deCode7(torch.cat((dec6, enc3), dim=1)) 
        dec8 = self.deCode8(torch.cat((dec7, enc2), dim=1))               
        out = self.lastLayer(torch.cat((dec8, enc1), dim=1))
        return out

class uNet4(nn.Module):
    def __init__(self, num_classes = 1 ):
        super(uNet4, self).__init__()
        self.enCode1 = convBlock(in_channels=1, out_channels=64)
        self.enCode2 = convBlock(in_channels=64, out_channels=64)
        self.enCode3 = convBlock(in_channels=64, out_channels=128)
        self.enCode4 = convBlock(in_channels=128, out_channels=256)
        self.Maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.deCode1 = upSampling(in_channels=256, middle_channels=128, out_channels=256)
        self.deCode2 = upSampling(in_channels=512, middle_channels=256, out_channels=128)
        self.deCode3 = upSampling(in_channels=256, middle_channels=128, out_channels=64)
        self.deCode4 = upSampling(in_channels=64*2, middle_channels=64, out_channels=64)     
        
        self.lastLayer = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv1d(64, num_classes, kernel_size=1, bias=False))  
        
    def forward(self, x):
        enc1 = self.enCode1(x)
        enc1_pool = self.Maxpool(enc1)
        enc2 = self.enCode2(enc1_pool)
        enc2_pool = self.Maxpool(enc2)
        enc3 = self.enCode3(enc2_pool)
        enc3_pool = self.Maxpool(enc3)
        enc4 = self.enCode4(enc3_pool)
        enc4_pool = self.Maxpool(enc4)
        dec1 = self.deCode1(enc4_pool)   
        dec2 = self.deCode2(torch.cat((dec1, enc4), dim=1))  
        dec3 = self.deCode3(torch.cat((dec2, enc3), dim=1))
        dec4 = self.deCode4(torch.cat((dec3, enc2), dim=1))                    
        out = self.lastLayer(torch.cat((dec4, enc1), dim=1))
        return out
    
def losss(spec_pre , spec_good, ):
    erro = torch.sqrt(torch.sum(torch.square(spec_pre - spec_good + 1e-8), 0))
    base = torch.sqrt(torch.sum(torch.square(spec_good), 0))
    loss = torch.mean(erro )
    return loss
def losss_stop(spec_pre , spec_good ):
    erro = torch.sqrt(torch.sum(torch.square(spec_pre-spec_good + 1e-8) ))
    base = torch.sqrt(torch.sum(torch.square(spec_good) ))
    loss = erro/base
    return loss
