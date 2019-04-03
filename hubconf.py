from ssd_PyT.src import model
from torch.utils.model_zoo import load_url


def nvidia_ssd_pyt(pretrained=True, *args, **kwargs):
    m = model.SSD300()
    if pretrained:
        checkpoint = 'http://kkudrynski-dt1.vpn.dyn.nvidia.com:5000/download/models/JoC_SSD_FP32_PyT-1ec96418.pt'
        ckpt = load_url(checkpoint, progress=False)
        m.load_state_dict(ckpt['model'])
    return m
