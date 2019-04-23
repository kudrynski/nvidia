from PyTorch.Detection.SSD.src import model as ssd
#from PyTorch.Recommendation.NCF import neumf as ncf
#from PyTorch.SpeechSynthesis.Tacotron2.waveglow import model as waveglow
#from PyTorch.SpeechSynthesis.Tacotron2.tacotron2 import model as tacotron2

import urllib.request

dependencies = ['torch']

def nvidia_ssd(pretrained=True, *args, **kwargs):
    """Constructs a SSD300 model. 
    For detailed information on model input and output, training recipies, inference and performance 
    visit: github.com/NVIDIA/DeepLearningExamples and ngc.nvidia.com

    Args:
        pretrained (bool): If True, returns a model pretrained on COCO dataset.
 
    """
    m = ssd.SSD300()
    if pretrained:
        checkpoint = 'http://kkudrynski-dt1.vpn.dyn.nvidia.com:5000/download/models/JoC_SSD_FP32_PyT.pt'
        ckpt_file = "ssd_ckpt.pt"
        urllib.request.urlretrieve(checkpoint, ckpt_file)
        ckpt = torch.load(ckpt_file)
        m.load_state_dict(ckpt['model'])
    return m

def nvidia_ncf(pretrained=True, *args, **kwargs):
    
    m = ncf.NeuMF(nb_users, nb_items, mf_dim, mf_reg, mlp_layer_sizes, mlp_layer_regs, dropout)
    if pretrained:
        checkpoint = 'http://kkudrynski-dt1.vpn.dyn.nvidia.com:5000/download/models/JoC_NCF_FP32_PyT.pt'
        ckpt_file = "ncf_ckpt.pt"
        urllib.request.urlretrieve(checkpoint, ckpt_file)
        ckpt = torch.load(ckpt_file)
        m.load_state_dict(ckpt['model'])
    return m


def nvidia_tacotron2(pretrained=True, *args, **kwargs):
    
    m = tacotron2.Tacotron2()
    if pretrained:
        checkpoint = 'http://kkudrynski-dt1.vpn.dyn.nvidia.com:5000/download/models/JoC_Tacotron2_FP32_PyT.pt'
        ckpt_file = "tacotron2_ckpt.pt"
        urllib.request.urlretrieve(checkpoint, ckpt_file)
        ckpt = torch.load(ckpt_file)
        m.load_state_dict(ckpt['model'])
    return m

def nvidia_waveglow(pretrained=True, *args, **kwargs):
    
    m = waveglow.WaveGlow()
    if pretrained:
        checkpoint = 'http://kkudrynski-dt1.vpn.dyn.nvidia.com:5000/download/models/JoC_WaveGlow_FP32_PyT.pt'
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
    inp = torch.randn([1,3,300,300], dtype=torch.float32)
    out = hub_model.forward(inp)
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
    ssd_test()

