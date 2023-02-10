import torch
import torch.nn as nn


def build_encoder(vgg_path=None):

    vgg = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1),
        nn.ReflectionPad2d(padding=1),
        nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3),
        nn.ReLU(), # relu1_1
        
        nn.ReflectionPad2d(padding=1),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),
        nn.ReLU(), # relu1_2
        
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.ReflectionPad2d(padding=1),
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),
        nn.ReLU(), # relu2_1
        
        nn.ReflectionPad2d(padding=1),
        nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3),
        nn.ReLU(), # relu2_2
        
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.ReflectionPad2d(padding=1),
        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3),
        nn.ReLU(), # relu3_1
        
        nn.ReflectionPad2d(padding=1),
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3),
        nn.ReLU(), # relu3_2
        
        nn.ReflectionPad2d(padding=1),
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3),
        nn.ReLU(), # relu3_3
        
        nn.ReflectionPad2d(padding=1),
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3),
        nn.ReLU(), # relu3_4
        
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.ReflectionPad2d(padding=1),
        nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3),
        nn.ReLU(), # relu4_1 (last layer of the encoder!)
        
        nn.ReflectionPad2d(padding=1),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3),
        nn.ReLU(), # relu4_2
        
        nn.ReflectionPad2d(padding=1),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3),
        nn.ReLU(), # relu4_3
        
        nn.ReflectionPad2d(padding=1),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3),
        nn.ReLU(), # relu4_4
        
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.ReflectionPad2d(padding=1),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3),
        nn.ReLU(), # relu5_1
        
        nn.ReflectionPad2d(padding=1),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3),
        nn.ReLU(), # relu5_2
        
        nn.ReflectionPad2d(padding=1),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3),
        nn.ReLU(), # relu5_3
        
        nn.ReflectionPad2d(padding=1),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3),
        nn.ReLU()  # relu5_4
    )
    
    if vgg_path is not None:
        vgg.load_state_dict(torch.load(vgg_path))
        
    encoder = nn.Sequential(*list(vgg.children())[:31])

    return encoder


def build_decoder(decoder_path=None):
    
    decoder = nn.Sequential(
        nn.ReflectionPad2d(padding=1),
        nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3),
        nn.ReLU(),
        
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.ReflectionPad2d(padding=1),
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3),
        nn.ReLU(),
        
        nn.ReflectionPad2d(padding=1),
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3),
        nn.ReLU(),
        
        nn.ReflectionPad2d(padding=1),
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3),
        nn.ReLU(),
        
        nn.ReflectionPad2d(padding=1),
        nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3),
        nn.ReLU(),
        
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.ReflectionPad2d(padding=1),
        nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3),
        nn.ReLU(),
        
        nn.ReflectionPad2d(padding=1),
        nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3),
        nn.ReLU(),
        
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.ReflectionPad2d(padding=1),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),
        nn.ReLU(),
        
        nn.ReflectionPad2d(padding=1),
        nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3),
    )
    
    if decoder_path is not None:
        decoder.load_state_dict(torch.load(decoder_path))
    
    return decoder


def get_mean_std(tensor, eps=1e-5):
    
    assert len(tensor.size()) == 4
    
    mean = torch.mean(tensor, dim=[2,3], keepdim=True)
    var  = torch.std(tensor, dim=[2,3], keepdim=True) + eps
    std  = torch.sqrt(var)

    return mean, std


class AdaIN(nn.Module):
    """ Implementation of Adaptive Instance Normalization (AdaIN) layer as described in the paper [Huang17]."""
    
    def __init__(self):
        super().__init__()

    def forward(self, content, style):
        content_mean, content_std = get_mean_std(content)
        style_mean,   style_std   = get_mean_std(style)
        return style_std * (content - content_mean) / content_std + style_mean