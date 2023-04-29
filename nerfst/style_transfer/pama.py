import torch
from nerfst.style_transfer.pama_model import PAMANet


def pama_infer_one_image(Ic, Is, hparams):
    model = PAMANet(hparams)
    model.eval()
    model.to('cuda')
    Ic = Ic.to('cuda')
    Is = Is.to('cuda')
    
    with torch.no_grad():
        Ics = model(Ic, Is)
    # ic(Ics.shape) #? 为啥图片会变大一点啊。 由下采样的时候宽高除不尽导致。

    return Ics
        
        
if __name__ == '__main__':
    ...