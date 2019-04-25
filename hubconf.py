import urllib.request


# from https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/SpeechSynthesis/Tacotron2/inference.py
def checkpoint_from_distributed(state_dict):
    """
    Checks whether checkpoint was generated by DistributedDataParallel. DDP
    wraps model in additional "module.", it needs to be unwrapped for single
    GPU inference.
    :param state_dict: model's state dict
    """
    ret = False
    for key, _ in state_dict.items():
        if key.find('module.') != -1:
            ret = True
            break
    return ret


# from https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/SpeechSynthesis/Tacotron2/inference.py
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


dependencies = ['torch']


def nvidia_ssd(pretrained=True, **kwargs):
    """Constructs an SSD300 model.
    For detailed information on model input and output, training recipies, inference and performance
    visit: github.com/NVIDIA/DeepLearningExamples and/or ngc.nvidia.com

    Args:
        pretrained (bool, True): If True, returns a model pretrained on COCO dataset.
    """

    from PyTorch.Detection.SSD.src import model as ssd

    m = ssd.SSD300()
    if pretrained:
        checkpoint = 'http://kkudrynski-dt1.vpn.dyn.nvidia.com:5000/download/models/JoC_SSD_FP32_PyT'
        ckpt_file = "ssd_ckpt.pt"
        urllib.request.urlretrieve(checkpoint, ckpt_file)
        ckpt = torch.load(ckpt_file)
        m.load_state_dict(ckpt['model'])
    return m


def nvidia_ncf(pretrained=True, **kwargs):
    """Constructs an NCF model.
    For detailed information on model input and output, training recipies, inference and performance
    visit: github.com/NVIDIA/DeepLearningExamples and/or ngc.nvidia.com

    Args:
        pretrained (bool, True): If True, returns a model pretrained on ml-20m dataset.
        nb_users (int): number of users
        nb_items (int): number of items
        mf_dim (int, 64): dimension of latent space in matrix factorization
        mlp_layer_sizes (list, [256,256,128,64]): sizes of layers of multi-layer-perceptron
        dropout (float, 0.5): dropout
    """

    from PyTorch.Recommendation.NCF import neumf as ncf

    config = {'nb_users':None, 'nb_items':None, 'mf_dim':64, 'mf_reg':0.,
              'mlp_layer_sizes':[256,256,128,64], 'mlp_layer_regs':[0,0,0,0], 'dropout':0.5}

    if pretrained:
        checkpoint = 'http://kkudrynski-dt1.vpn.dyn.nvidia.com:5000/download/models/JoC_NCF_FP32_PyT'
        ckpt_file = "ncf_ckpt.pt"
        urllib.request.urlretrieve(checkpoint, ckpt_file)
        ckpt = torch.load(ckpt_file)

        if checkpoint_from_distributed(ckpt):
            ckpt = unwrap_distributed(ckpt)

        config['nb_users'] = ckpt['mf_user_embed.weight'].shape[0]
        config['nb_items'] = ckpt['mf_item_embed.weight'].shape[0]
        config['mf_dim'] = ckpt['mf_item_embed.weight'].shape[1]
        mlp_shapes = [ckpt[k].shape for k in ckpt.keys() if 'mlp' in k and 'weight' in k and 'embed' not in k]
        config['mlp_layer_sizes'] = [mlp_shapes[0][1], mlp_shapes[1][1], mlp_shapes[2][1],  mlp_shapes[2][0]]
        config['mlp_layer_regs'] =  [0] * len(config['mlp_layer_sizes'])

        m = ncf.NeuMF(**config)
        m.load_state_dict(ckpt)

    else:
        if 'nb_users' not in kwargs:
            raise ValueError("Missing 'nb_users' argument.")
        if 'nb_items' not in kwargs:
            raise ValueError("Missing 'nb_items' argument.")
        for k,v in kwargs.items():
            if k in config.keys():
                config[k] = v
        config['mlp_layer_regs'] =  [0] * len(config['mlp_layer_sizes'])
        m = ncf.NeuMF(**config)

    return m


def nvidia_tacotron2(pretrained=True, **kwargs):
    """Constructs a Tacotron 2 model (nn.module with additional inference(input) method).
    For detailed information on model input and output, training recipies, inference and performance
    visit: github.com/NVIDIA/DeepLearningExamples and/or ngc.nvidia.com

    Args (type[, default value]):
        pretrained (bool, True): If True, returns a model pretrained on LJ Speech dataset.
        n_symbols (int, 148): Number of symbols used in a sequence passed to the prenet, see
                              https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/SpeechSynthesis/Tacotron2/tacotron2/text/symbols.py
        p_attention_dropout (float, 0.1): dropout probability on attention LSTM (1st LSTM layer in decoder)
        p_decoder_dropout (float, 0.1): dropout probability on decoder LSTM (2nd LSTM layer in decoder)
        max_decoder_steps (int, 1000): maximum number of generated mel spectrograms during inference
    """

    from PyTorch.SpeechSynthesis.Tacotron2.tacotron2 import model as tacotron2
    from PyTorch.SpeechSynthesis.Tacotron2.models import lstmcell_to_float, batchnorm_to_float

    if pretrained:
        checkpoint = 'http://kkudrynski-dt1.vpn.dyn.nvidia.com:5000/download/models/JoC_Tacotron2_FP32_PyT'
        ckpt_file = "tacotron2_ckpt.pt"
        urllib.request.urlretrieve(checkpoint, ckpt_file)
        ckpt = torch.load(ckpt_file)
        state_dict = ckpt['state_dict']
        if checkpoint_from_distributed(state_dict):
            state_dict = unwrap_distributed(state_dict)
        config = ckpt['config']
        m = tacotron2.Tacotron2(**config)
        m.load_state_dict(state_dict)
    else:
        config = {'mask_padding': False, 'n_mel_channels': 80, 'n_symbols': 148,
                  'symbols_embedding_dim': 512, 'encoder_kernel_size': 5,
                  'encoder_n_convolutions': 3, 'encoder_embedding_dim': 512,
                  'attention_rnn_dim': 1024, 'attention_dim': 128,
                  'attention_location_n_filters': 32,
                  'attention_location_kernel_size': 31, 'n_frames_per_step': 1,
                  'decoder_rnn_dim': 1024, 'prenet_dim': 256,
                  'max_decoder_steps': 1000, 'gate_threshold': 0.5,
                  'p_attention_dropout': 0.1, 'p_decoder_dropout': 0.1,
                  'postnet_embedding_dim': 512, 'postnet_kernel_size': 5,
                  'postnet_n_convolutions': 5, 'decoder_no_early_stopping': False}

        for k,v in kwargs.items():
            if k in config.keys():
                config[k] = v
        m = tacotron2.Tacotron2(**config)

    for k,v in kwargs.items():
        if k == "model_math" and v == "fp16":
            m = batchnorm_to_float(m.half())
            print("for fp16 training, use `model = lstmcell_to_float(model)`")
    return m


def nvidia_waveglow(pretrained=True, **kwargs):
    """Constructs a WaveGlow model (nn.module with additional infer(input) method).
    For detailed information on model input and output, training recipies, inference and performance
    visit: github.com/NVIDIA/DeepLearningExamples and/or ngc.nvidia.com

    Args:
        pretrained (bool): If True, returns a model pretrained on LJ Speech dataset.
    """

    from PyTorch.SpeechSynthesis.Tacotron2.waveglow import model as waveglow
    from PyTorch.SpeechSynthesis.Tacotron2.models import batchnorm_to_float
    if pretrained:
        checkpoint = 'http://kkudrynski-dt1.vpn.dyn.nvidia.com:5000/download/models/JoC_WaveGlow_FP32_PyT'
        ckpt_file = "waveglow_ckpt.pt"
        urllib.request.urlretrieve(checkpoint, ckpt_file)
        ckpt = torch.load(ckpt_file)
        state_dict = ckpt['state_dict']
        if checkpoint_from_distributed(state_dict):
            state_dict = unwrap_distributed(state_dict)
        config = ckpt['config']
        m = waveglow.WaveGlow(**config)
        m.load_state_dict(state_dict)
    else:
        config = {'n_mel_channels': 80, 'n_flows': 12, 'n_group': 8,
                  'n_early_every': 4, 'n_early_size': 2,
                  'WN_config': {'n_layers': 8, 'kernel_size': 3,
                                'n_channels': 512}}
        for k,v in kwargs.items():
            if k in config.keys():
                config[k] = v
            elif k in config['WN_config'].keys():
                config['WN_config'][k] = v
        m = waveglow.WaveGlow(**config)

    for k,v in kwargs.items():
        if k == "model_math" and v == "fp16":
            m = batchnorm_to_float(m.half())
            for k in m.convinv:
                k.float()
    return m

# temporary tests:


import torch


def ssd_test():
    print('\nssd test output')
    hub_model = nvidia_ssd().cuda()
    hub_model.eval()
    inp = torch.randn([1,3,300,300], dtype=torch.float32).cuda()
    with torch.no_grad():
        out = hub_model.forward(inp)
    print(out[0].size())
    print(out[1].size())


def ncf_test(**kwargs):
    print('\nncf test output')
    hub_model = nvidia_ncf(**kwargs).cuda()
    hub_model.eval()
    input_users=torch.tensor([0,1,2]).cuda()
    input_items=torch.tensor([0,1,2]).cuda()
    with torch.no_grad():
        out = hub_model(input_users, input_items, sigmoid=True)
    print(out.size())


def tacotron2_test():
    print('\ntacotron2 test output')
    model_math = "fp16"
    hub_model = nvidia_tacotron2(model_math=model_math)
    hub_model = hub_model.cuda()
    hub_model.eval()
    inp = torch.randint(low=0, high=148, size=(1,140), dtype=torch.long)
    inp = torch.autograd.Variable(inp).cuda().long()
    with torch.no_grad():
        _, mel, _, _ = hub_model.infer(inp)
    print(mel.size())


def waveglow_test():
    print('\nwaveglow test output')
    model_math = "fp16"
    hub_model = nvidia_waveglow(model_math=model_math)
    hub_model = hub_model.cuda()
    hub_model = hub_model.remove_weightnorm(hub_model)
    hub_model.eval()
    inp = torch.randn([1,80,300], dtype=torch.float32).cuda()
    if model_math == "fp16":
        inp = inp.half()
    with torch.no_grad():
        out = hub_model.infer(inp)
    print(out.size())


if __name__ == '__main__':
    ssd_test()
    ncf_test()
    ncf_test(pretrained=False, nb_users=100, nb_items=100)
    tacotron2_test()
    waveglow_test()
