import torch

# from facelib.utils import load_file_from_url
# from .bisenet import BiSeNet
from .parsenet import ParseNet
from config import MODEL_FACE_LIB

def init_parsing_model(model_name='bisenet', half=False, device='cuda'):
    # if model_name == 'bisenet':
    #     model = BiSeNet(num_class=19)
    #     model_url = 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/parsing_bisenet.pth'
    # elif model_name == 'parsenet':
    #     model = ParseNet(in_size=512, out_size=512, parsing_ch=19)
    #     model_url = 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/parsing_parsenet.pth'
    # else:
    #     raise NotImplementedError(f'{model_name} is not implemented.')

    # model_path = load_file_from_url(url=model_url, model_dir='model/weights/facelib', progress=True, file_name=None)
    if model_name == 'parsenet':
        model = ParseNet(in_size=512, out_size=512, parsing_ch=19)
        model_path = MODEL_FACE_LIB
    else:
        raise NotImplementedError(f'{model_name} is not implemented.')
    load_net = torch.load(model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(load_net, strict=True)
    model.eval()
    model = model.to(device)
    return model
