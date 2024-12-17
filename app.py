import gradio as gr 
from PIL import Image
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F

from archs import DarkIR



device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#define some auxiliary functions
pil_to_tensor = transforms.ToTensor()
tensor_to_pil = transforms.ToPILImage()

network = 'DarkIR'

PATH_MODEL = './models/darkir_1k_allv2_251205.pt'

model = DarkIR(img_channel=3, 
                    width=32, 
                    middle_blk_num_enc=2,
                    middle_blk_num_dec=2, 
                    enc_blk_nums=[1, 2, 3],
                    dec_blk_nums=[3, 1, 1], 
                    dilations=[1, 4, 9],
                    extra_depth_wise=True)

checkpoints = torch.load(PATH_MODEL, map_location=device)
model.load_state_dict(checkpoints['params'])

model = model.to(device)

def path_to_tensor(path):
    img = Image.open(path).convert('RGB')
    img = pil_to_tensor(img).unsqueeze(0)
    
    return img
def normalize_tensor(tensor):
    
    max_value = torch.max(tensor)
    min_value = torch.min(tensor)
    output = (tensor - min_value)/(max_value)
    return output

def pad_tensor(tensor, multiple = 8):
    '''pad the tensor to be multiple of some number'''
    multiple = multiple
    _, _, H, W = tensor.shape
    pad_h = (multiple - H % multiple) % multiple
    pad_w = (multiple - W % multiple) % multiple
    tensor = F.pad(tensor, (0, pad_w, 0, pad_h), value = 0)
    
    return tensor

def process_img(image):
    tensor = path_to_tensor(image).to(device)
    _, _, H, W = tensor.shape
    
    tensor = pad_tensor(tensor)

    with torch.no_grad():
        output = model(tensor, side_loss=False)

    output = torch.clamp(output, 0., 1.)
    output = output[:,:, :H, :W].squeeze(0)    
    return tensor_to_pil(output)

title = "DarkIR ✏️🖼️ 🤗"
description = ''' ## [ DarkIR: Robust Low-Light Image Restoration](https://github.com/cidautai/DarkIR)

[Daniel Feijoo](https://github.com/danifei)

Fundación Cidaut


> **Disclaimer:** please remember this is not a product, thus, you will notice some limitations.
**This demo expects an image with some Low-Light degradations.**

<br>
'''

examples = [['examples/0010.png'],
            ['examples/r13073518t_low.png'], 
            ['examples/low00733_low.png'], 
            ["examples/0087.png"]]

css = """
    .image-frame img, .image-container img {
        width: auto;
        height: auto;
        max-width: none;
    }
"""

demo = gr.Interface(
    fn = process_img,
    inputs = [
            gr.Image(type = 'pil', label = 'input')
    ],
    outputs = [gr.Image(type='pil', label = 'output')],
    title = title,
    description = description,
    examples = examples,
    css = css
)

if __name__ == '__main__':
    demo.launch()