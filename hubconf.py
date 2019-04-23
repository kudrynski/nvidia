from PyTorch.Detection.SSD.src import model as ssd
from PyTorch.Recommendation.NCF import neumf as ncf
#from PyTorch.SpeechSynthesis.Tacotron2.waveglow import model as waveglow
#from PyTorch.SpeechSynthesis.Tacotron2.tacotron2 import model as tacotron2

import urllib.request

dependencies = ['torch']

def nvidia_ssd(pretrained=True, *args, **kwargs):
    """Constructs an SSD300 model. 
    For detailed information on model input and output, training recipies, inference and performance 
    visit: github.com/NVIDIA/DeepLearningExamples and/or ngc.nvidia.com

    Args:
        pretrained (bool): If True, returns a model pretrained on COCO dataset.
 
    """
    m = ssd.SSD300()
    if pretrained:
        checkpoint = 'http://kkudrynski-dt1.vpn.dyn.nvidia.com:5000/download/models/JoC_SSD_FP32_PyT'
        ckpt_file = "ssd_ckpt.pt"
        urllib.request.urlretrieve(checkpoint, ckpt_file)
        ckpt = torch.load(ckpt_file)
        m.load_state_dict(ckpt['model'])
    return m


def unwrap_distributed(state_dict):
    """
    Unwraps model from DistributedDataParallel.
    DDP wraps model in additional "module.", it needs to be removed for single
    GPU inference.
    :param state_dict: model's state dict
    """
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('module.', '')
        new_state_dict[new_key] = value
    return new_state_dict

def nvidia_ncf(pretrained=True, *args, **kwargs):
    """Constructs an NCF model. 
    For detailed information on model input and output, training recipies, inference and performance 
    visit: github.com/NVIDIA/DeepLearningExamples and/or ngc.nvidia.com

    Args:
        pretrained (bool): If True, returns a model pretrained on ml-20m dataset.
 
    """
    if pretrained:
        checkpoint = 'http://kkudrynski-dt1.vpn.dyn.nvidia.com:5000/download/models/JoC_NCF_FP32_PyT'
        ckpt_file = "ncf_ckpt.pt"
        urllib.request.urlretrieve(checkpoint, ckpt_file)
        ckpt = torch.load(ckpt_file)
       
        ckpt = unwrap_distributed(ckpt)

        nb_users = ckpt['mf_user_embed.weight'].shape[0]
        nb_items = ckpt['mf_item_embed.weight'].shape[0]
        mf_dim = ckpt['mf_item_embed.weight'].shape[1]
        mf_reg = 0.
        mlp_shapes = [ckpt[k].shape for k in ckpt.keys() if 'mlp' in k and 'weight' in k and 'embed' not in k]
        mlp_layer_sizes = [mlp_shapes[0][1], mlp_shapes[1][1], mlp_shapes[2][1],  mlp_shapes[2][0]]
        mlp_layer_regs =  [0] * len(mlp_layer_sizes)
        dropout = 0.5

        m = ncf.NeuMF(nb_users, nb_items, mf_dim, mf_reg, mlp_layer_sizes, mlp_layer_regs, dropout)
        m.load_state_dict(ckpt)
    else:
        pass
        #m = ncf.NeuMF(nb_users, nb_items, mf_dim, mf_reg, mlp_layer_sizes, mlp_layer_regs, dropout)
    
    return m


def nvidia_tacotron2(pretrained=True, *args, **kwargs):
    
    m = tacotron2.Tacotron2()
    if pretrained:
        checkpoint = 'http://kkudrynski-dt1.vpn.dyn.nvidia.com:5000/download/models/JoC_Tacotron2_FP32_PyT'
        ckpt_file = "tacotron2_ckpt.pt"
        urllib.request.urlretrieve(checkpoint, ckpt_file)
        ckpt = torch.load(ckpt_file)
        m.load_state_dict(ckpt['model'])
    return m

def nvidia_waveglow(pretrained=True, *args, **kwargs):
    
    m = waveglow.WaveGlow()
    if pretrained:
        checkpoint = 'http://kkudrynski-dt1.vpn.dyn.nvidia.com:5000/download/models/JoC_WaveGlow_FP32_PyT'
        ckpt_file = "waveglow_ckpt.pt"
        urllib.request.urlretrieve(checkpoint, ckpt_file)
        ckpt = torch.load(ckpt_file)
        m.load_state_dict(ckpt['model'])
    return m


# temporary tests:

import torch

def ssd_test():
    hub_model = nvidia_ssd()
    hub_model.eval()
    inp = torch.randn([1,3,300,300], dtype=torch.float32)
    out = hub_model.forward(inp)
    print(out)


def ncf_test():
    hub_model = nvidia_ncf()
    hub_model.eval()
    out = hub_model(torch.tensor([0,1,2]),torch.tensor([0,1,2]), sigmoid=True)
    print(out)


def tacotron2_test():
    hub_model = nvidia_tacotron2()
    hub_model.eval()
    inp = torch.randn([1,3,300,300], dtype=torch.float32)
    out = hub_model.forward(inp)
    print(out)


def waveglow_test():
    hub_model = nvidia_waveglow()
    hub_model.eval()
    inp = torch.randn([1,3,300,300], dtype=torch.float32)
    out = hub_model.forward(inp)
    print(out)


   
if __name__ == '__main__':
    #ssd_test()
    ncf_test()

