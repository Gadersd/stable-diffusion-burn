use super::GroupNorm;
use crate::model::load::*;

use std::error::Error;

use burn::{
    config::Config,
    module::{Module, Param},
    nn,
    tensor::{backend::Backend, Tensor},
};

use super::*;
use crate::model::groupnorm::load::load_group_norm;

fn load_conv_self_attention_block<B: Backend>(
    path: &str,
    device: &B::Device,
) -> Result<ConvSelfAttentionBlock<B>, Box<dyn Error>> {
    let norm = load_group_norm(&format!("{}/{}", path, "norm"), device)?;
    let q = load_conv2d(&format!("{}/{}", path, "q"), device)?;
    let k = load_conv2d(&format!("{}/{}", path, "k"), device)?;
    let v = load_conv2d(&format!("{}/{}", path, "v"), device)?;
    let proj_out = load_conv2d(&format!("{}/{}", path, "proj_out"), device)?;

    Ok(ConvSelfAttentionBlock {
        norm,
        q,
        k,
        v,
        proj_out,
    })
}

fn load_resnet_block<B: Backend>(
    path: &str,
    device: &B::Device,
) -> Result<ResnetBlock<B>, Box<dyn Error>> {
    let norm1 = load_group_norm(&format!("{}/{}", path, "norm1"), device)?;
    let silu1 = SILU {};
    let conv1 = load_conv2d(&format!("{}/{}", path, "conv1"), device)?;
    let norm2 = load_group_norm(&format!("{}/{}", path, "norm2"), device)?;
    let silu2 = SILU {};
    let conv2 = load_conv2d(&format!("{}/{}", path, "conv2"), device)?;
    let nin_shortcut = load_conv2d(&format!("{}/{}", path, "nin_shortcut"), device).ok();

    Ok(ResnetBlock {
        norm1,
        silu1,
        conv1,
        norm2,
        silu2,
        conv2,
        nin_shortcut,
    })
}

fn load_mid<B: Backend>(path: &str, device: &B::Device) -> Result<Mid<B>, Box<dyn Error>> {
    let block_1 = load_resnet_block(&format!("{}/{}", path, "block_1"), device)?;
    let attn = load_conv_self_attention_block(&format!("{}/{}", path, "attn"), device)?;
    let block_2 = load_resnet_block(&format!("{}/{}", path, "block_2"), device)?;

    Ok(Mid {
        block_1,
        attn,
        block_2,
    })
}

fn load_padded_conv2d<B: Backend>(
    path: &str,
    device: &B::Device,
) -> Result<PaddedConv2d<B>, Box<dyn Error>> {
    let mut conv = load_conv2d(&format!("{}/{}", path, "conv"), device)?;

    let channels = load_tensor::<B, 1>("channels", path, device)?;
    let channels = tensor_to_array_2(channels);

    let kernel_size = load_usize::<B>("kernel_size", path, device)?;
    let stride = load_usize::<B>("stride", path, device)?;

    let padding = load_tensor::<B, 1>("padding", path, device)?;
    let padding: [usize; 4] = tensor_to_array(padding);
    let padding = PaddingCfg::new(padding[0], padding[1], padding[2], padding[3]);

    //let mut record = conv.into_record();

    let mut padded_conv: PaddedConv2d<B> = PaddedConv2dConfig::new(channels, kernel_size, padding)
        .with_stride(stride)
        .init(device);
    let padding_actual =
        PaddingConfig2d::Explicit(padded_conv.padding_actual[0], padded_conv.padding_actual[1]);

    conv.padding =  burn::module::Ignored(padding_actual);
    padded_conv.conv = conv;

    //record.padding = <PaddingConfig2d as Module<B>>::into_record(padding_actual);
    //padded_conv.conv = padded_conv.conv.load_record(record);

    Ok(padded_conv)
}

fn load_decoder_block<B: Backend>(
    path: &str,
    device: &B::Device,
) -> Result<DecoderBlock<B>, Box<dyn Error>> {
    let res1 = load_resnet_block(&format!("{}/{}", path, "res1"), device)?;
    let res2 = load_resnet_block(&format!("{}/{}", path, "res2"), device)?;
    let res3 = load_resnet_block(&format!("{}/{}", path, "res3"), device)?;
    let upsampler = load_conv2d(&format!("{}/{}", path, "upsampler"), device).ok();

    Ok(DecoderBlock {
        res1,
        res2,
        res3,
        upsampler,
    })
}

fn load_encoder_block<B: Backend>(
    path: &str,
    device: &B::Device,
) -> Result<EncoderBlock<B>, Box<dyn Error>> {
    let res1 = load_resnet_block(&format!("{}/{}", path, "res1"), device)?;
    let res2 = load_resnet_block(&format!("{}/{}", path, "res2"), device)?;
    let downsampler = load_padded_conv2d(&format!("{}/{}", path, "downsampler"), device).ok();

    Ok(EncoderBlock {
        res1,
        res2,
        downsampler,
    })
}

fn load_decoder<B: Backend>(path: &str, device: &B::Device) -> Result<Decoder<B>, Box<dyn Error>> {
    let conv_in = load_conv2d(&format!("{}/{}", path, "conv_in"), device)?;
    let mid = load_mid(&format!("{}/{}", path, "mid"), device)?;

    let n_block = load_usize::<B>("n_block", path, device)?;
    let mut blocks = (0..n_block)
        .into_iter()
        .map(|i| load_decoder_block::<B>(&format!("{}/blocks/{}", path, i), device))
        .collect::<Result<Vec<_>, _>>()?;

    let norm_out = load_group_norm(&format!("{}/{}", path, "norm_out"), device)?;
    let silu = SILU {};
    let conv_out = load_conv2d(&format!("{}/{}", path, "conv_out"), device)?;

    Ok(Decoder {
        conv_in,
        mid,
        blocks,
        norm_out,
        silu,
        conv_out,
    })
}

fn load_encoder<B: Backend>(path: &str, device: &B::Device) -> Result<Encoder<B>, Box<dyn Error>> {
    let conv_in = load_conv2d(&format!("{}/{}", path, "conv_in"), device)?;
    let mid = load_mid(&format!("{}/{}", path, "mid"), device)?;

    let n_block = load_usize::<B>("n_block", path, device)?;
    let mut blocks = (0..n_block)
        .into_iter()
        .map(|i| load_encoder_block::<B>(&format!("{}/blocks/{}", path, i), device))
        .collect::<Result<Vec<_>, _>>()?;

    let norm_out = load_group_norm(&format!("{}/{}", path, "norm_out"), device)?;
    let silu = SILU {};
    let conv_out = load_conv2d(&format!("{}/{}", path, "conv_out"), device)?;

    Ok(Encoder {
        conv_in,
        mid,
        blocks,
        norm_out,
        silu,
        conv_out,
    })
}

pub fn load_autoencoder<B: Backend>(
    path: &str,
    device: &B::Device,
) -> Result<Autoencoder<B>, Box<dyn Error>> {
    let encoder = load_encoder(&format!("{}/{}", path, "encoder"), device)?;
    let decoder = load_decoder(&format!("{}/{}", path, "decoder"), device)?;
    let quant_conv = load_conv2d(&format!("{}/{}", path, "quant_conv"), device)?;
    let post_quant_conv = load_conv2d(&format!("{}/{}", path, "post_quant_conv"), device)?;

    Ok(Autoencoder {
        encoder,
        decoder,
        quant_conv,
        post_quant_conv,
    })
}
