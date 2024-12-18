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

PATH_MODEL = './DarkIR_384.pt'

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


def pad_tensor(tensor, multiple = 8):
    '''pad the tensor to be multiple of some number'''
    multiple = multiple
    _, _, H, W = tensor.shape
    pad_h = (multiple - H % multiple) % multiple
    pad_w = (multiple - W % multiple) % multiple
    tensor = F.pad(tensor, (0, pad_w, 0, pad_h), value = 0)
    
    return tensor

def process_img(image):
    tensor = pil_to_tensor(image).unsqueeze(0).to(device)
    _, _, H, W = tensor.shape
    
    tensor = pad_tensor(tensor)

    with torch.no_grad():
        output = model(tensor, side_loss=False)

    output = torch.clamp(output, 0., 1.)
    output = output[:,:, :H, :W].squeeze(0)    
    return tensor_to_pil(output)

title = "DarkIR ✏️🖼️ 🤗"
description = ''' ## [ DarkIR: Robust Low-Light Image Restoration](https://github.com/cidautai/DarkIR)

[Daniel Feijoo](https://github.com/danifei), Juan C. Benito, Álvaro García, Marcos V. Conde

[Fundación Cidaut](https://cidaut.ai/)

>**Abstract**:Photography during night or in dark conditions typically suffers from noise, low light and blurring issues due to the dim environment and the common use of long exposure. Although Deblurring and Low-light Image Enhancement (LLIE) are related under these conditions, most approaches in image restoration solve these tasks separately. In this paper, we present an efficient and robust neural network for multi-task low-light image restoration. Instead of following the current tendency of Transformer-based models, we propose new attention mechanisms to enhance the receptive field of efficient CNNs. Our method reduces the computational costs in terms of parameters and MAC operations compared to previous methods. Our model, DarkIR, achieves new state-of-the-art results on the popular LOLBlur, LOLv2 and Real-LOLBlur datasets, being able to generalize on real-world night and dark images.

Available code at [github](https://github.com/cidautai/DarkIR). More information on the [Arxiv paper](). 

> **Disclaimer:** please remember this is not a product, thus, you will notice some limitations.
**This demo expects an image with some Low-Light degradations.**

<br>
'''

examples = [['examples/0010.png'],
            ['examples/1001.png'],
            ['examples/1100.png'], 
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