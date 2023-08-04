import pathlib
import save
from save import *

from tinygrad.nn import Conv2d

def save_resnet_block(resnet_block, path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    
    save_group_norm(resnet_block.norm1, pathlib.Path(path, 'norm1'))
    save_conv2d(resnet_block.conv1, pathlib.Path(path, 'conv1'))
    save_group_norm(resnet_block.norm2, pathlib.Path(path, 'norm2'))
    save_conv2d(resnet_block.conv2, pathlib.Path(path, 'conv2'))

    if isinstance(resnet_block.nin_shortcut, Conv2d):
        save_conv2d(resnet_block.nin_shortcut, pathlib.Path(path, 'nin_shortcut'))

def save_attn_block(attn_block, path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    
    save_group_norm(attn_block.norm, pathlib.Path(path, 'norm'))
    save_conv2d(attn_block.q, pathlib.Path(path, 'q'))
    save_conv2d(attn_block.k, pathlib.Path(path, 'k'))
    save_conv2d(attn_block.v, pathlib.Path(path, 'v'))
    save_conv2d(attn_block.proj_out, pathlib.Path(path, 'proj_out'))

def save_mid(mid, path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)

    save_resnet_block(mid.block_1, pathlib.Path(path, 'block_1'))
    save_attn_block(mid.attn_1, pathlib.Path(path, 'attn'))
    save_resnet_block(mid.block_2, pathlib.Path(path, 'block_2'))

def save_decoder_block(decoder_block, path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    
    if 'block' in decoder_block:
        save_resnet_block(decoder_block["block"][0], pathlib.Path(path, 'res1'))
        save_resnet_block(decoder_block["block"][1], pathlib.Path(path, 'res2'))
        save_resnet_block(decoder_block["block"][2], pathlib.Path(path, 'res3'))

    if 'upsample' in decoder_block:
        save_conv2d(decoder_block['upsample']['conv'], pathlib.Path(path, 'upsampler'))


def save_decoder(decoder, path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    
    save_conv2d(decoder.conv_in, pathlib.Path(path, 'conv_in'))
    save_mid(decoder.mid, pathlib.Path(path, 'mid'))

    for i, block in enumerate(decoder.up[::-1]):
        print(i)
        if isinstance(block['block'][0].nin_shortcut, Conv2d):
            print(block['block'][0].nin_shortcut.weight.shape)
        save_decoder_block(block, pathlib.Path(path, f'blocks/{i}'))

    save_scalar(len(decoder.up), "n_block", path)
    save_group_norm(decoder.norm_out, pathlib.Path(path, 'norm_out'))
    save_conv2d(decoder.conv_out, pathlib.Path(path, 'conv_out'))

def save_encoder_block(encoder_block, path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)

    if 'block' in encoder_block:
        save_resnet_block(encoder_block["block"][0], pathlib.Path(path, 'res1'))
        save_resnet_block(encoder_block["block"][1], pathlib.Path(path, 'res2'))
  
    if 'downsample' in encoder_block:
        save_padded_conv2d(encoder_block['downsample']['conv'], pathlib.Path(path, 'downsampler'))

def save_encoder(encoder, path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    
    save_conv2d(encoder.conv_in, pathlib.Path(path, 'conv_in'))
    save_mid(encoder.mid, pathlib.Path(path, 'mid'))

    for i, block in enumerate(encoder.down):
        save_encoder_block(block, pathlib.Path(path, f'blocks/{i}'))

    save_scalar(len(encoder.down), "n_block", path)
    save_group_norm(encoder.norm_out, pathlib.Path(path, 'norm_out'))
    save_conv2d(encoder.conv_out, pathlib.Path(path, 'conv_out'))
    

def save_autoencoder(autoencoder, path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)

    save_encoder(autoencoder.encoder, pathlib.Path(path, 'encoder'))
    save_decoder(autoencoder.decoder, pathlib.Path(path, 'decoder'))
    save_conv2d(autoencoder.quant_conv, pathlib.Path(path, 'quant_conv'))
    save_conv2d(autoencoder.post_quant_conv, pathlib.Path(path, 'post_quant_conv'))