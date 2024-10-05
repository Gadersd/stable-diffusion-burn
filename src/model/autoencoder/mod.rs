pub mod load;

use burn::{
    config::Config,
    module::{Module, Param},
    nn::{
        self,
        conv::{Conv2d, Conv2dConfig, Conv2dRecord},
        PaddingConfig2d,
    },
    tensor::{
        activation::{sigmoid, softmax},
        backend::Backend,
        module::embedding,
        Distribution, Int, Tensor,
    },
};

use super::groupnorm::*;
use super::silu::*;
//use crate::backend::Backend as MyBackend;
use crate::backend::{qkv_attention, attn_decoder_mask};

use std::iter;

#[derive(Config)]
pub struct AutoencoderConfig {}

impl AutoencoderConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Autoencoder<B> {
        let encoder =
            EncoderConfig::new(vec![(128, 128), (128, 256), (256, 512), (512, 512)], 32, 8).init(device);
        let decoder =
            DecoderConfig::new(vec![(512, 512), (512, 512), (512, 256), (256, 128)], 32).init(device);
        let quant_conv = Conv2dConfig::new([8, 8], [1, 1]).init(device);
        let post_quant_conv = Conv2dConfig::new([4, 4], [1, 1]).init(device);

        Autoencoder {
            encoder,
            decoder,
            quant_conv,
            post_quant_conv,
        }
    }
}

#[derive(Module, Debug)]
pub struct Autoencoder<B: Backend> {
    encoder: Encoder<B>,
    decoder: Decoder<B>,
    quant_conv: Conv2d<B>,
    post_quant_conv: Conv2d<B>,
}

impl<B: Backend> Autoencoder<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        self.decode_latent(self.encode_image(x))
    }

    pub fn encode_image(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let [n_batch, _, _, _] = x.dims();
        let latent = self.encoder.forward(x);
        let latent = self.quant_conv.forward(latent);
        let latent = latent.slice([0..n_batch, 0..4]);
        latent
    }

    pub fn decode_latent(&self, latent: Tensor<B, 4>) -> Tensor<B, 4> {
        let latent = self.post_quant_conv.forward(latent);
        self.decoder.forward(latent)
    }
}

#[derive(Config)]
pub struct EncoderConfig {
    channels: Vec<(usize, usize)>,
    n_group: usize,
    n_channels_out: usize,
}

impl EncoderConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> Encoder<B> {
        let n_expanded_channels_initial = self
            .channels
            .first()
            .map(|f| f.1)
            .expect("Channels must not be empty.");
        let n_expanded_channels_final = self.channels.first().unwrap().0;

        let conv_in = Conv2dConfig::new([3, n_expanded_channels_initial], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);

        let blocks = self
            .channels
            .iter()
            .enumerate()
            .map(|(i, &(n_channel_in, n_channel_out))| {
                let downsample = i != self.channels.len() - 1;
                EncoderBlockConfig::new(n_channel_in, n_channel_out, downsample).init(device)
            })
            .collect();

        let mid = MidConfig::new(n_expanded_channels_final).init(device);
        let norm_out = GroupNormConfig::new(self.n_group, n_expanded_channels_final).init(device);
        let silu = SILU::new();
        let conv_out = Conv2dConfig::new([n_expanded_channels_final, self.n_channels_out], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);

        Encoder {
            conv_in,
            mid,
            blocks,
            norm_out,
            silu,
            conv_out,
        }
    }
}

#[derive(Module, Debug)]
pub struct Encoder<B: Backend> {
    conv_in: Conv2d<B>,
    mid: Mid<B>,
    blocks: Vec<EncoderBlock<B>>,
    norm_out: GroupNorm<B>,
    silu: SILU,
    conv_out: Conv2d<B>,
}

impl<B: Backend> Encoder<B> {
    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.conv_in.forward(x);

        let mut x = x;
        for block in &self.blocks {
            x = block.forward(x);
        }

        let x = self.mid.forward(x);
        self.conv_out
            .forward(self.silu.forward(self.norm_out.forward(x)))
    }
}

#[derive(Config)]
pub struct DecoderConfig {
    channels: Vec<(usize, usize)>,
    n_group: usize,
}

impl DecoderConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> Decoder<B> {
        let n_expanded_channels = self
            .channels
            .first()
            .map(|f| f.0)
            .expect("Channels must not be empty.");
        let n_condensed_channels = self.channels.last().unwrap().1;

        let conv_in = Conv2dConfig::new([4, n_expanded_channels], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);
        let mid = MidConfig::new(n_expanded_channels).init(device);

        let blocks = self
            .channels
            .iter()
            .enumerate()
            .map(|(i, &(n_channel_in, n_channel_out))| {
                let upsample = i != self.channels.len() - 1;
                DecoderBlockConfig::new(n_channel_in, n_channel_out, upsample).init(device)
            })
            .collect();

        let norm_out = GroupNormConfig::new(self.n_group, n_condensed_channels).init(device);
        let silu = SILU::new();
        let conv_out = Conv2dConfig::new([n_condensed_channels, 3], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);

        Decoder {
            conv_in,
            mid,
            blocks,
            norm_out,
            silu,
            conv_out,
        }
    }
}

#[derive(Module, Debug)]
pub struct Decoder<B: Backend> {
    conv_in: Conv2d<B>,
    mid: Mid<B>,
    blocks: Vec<DecoderBlock<B>>,
    norm_out: GroupNorm<B>,
    silu: SILU,
    conv_out: Conv2d<B>,
}

impl<B: Backend> Decoder<B> {
    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.conv_in.forward(x);
        let x = self.mid.forward(x);

        let mut x = x;
        for block in &self.blocks {
            x = block.forward(x);
        }

        self.conv_out
            .forward(self.silu.forward(self.norm_out.forward(x)))
    }
}

#[derive(Config)]
pub struct EncoderBlockConfig {
    n_channels_in: usize,
    n_channels_out: usize,
    downsample: bool,
}

impl EncoderBlockConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> EncoderBlock<B> {
        let res1 = ResnetBlockConfig::new(self.n_channels_in, self.n_channels_out).init(device);
        let res2 = ResnetBlockConfig::new(self.n_channels_out, self.n_channels_out).init(device);
        let downsampler = if self.downsample {
            let padding = PaddingCfg::new(0, 1, 0, 1);
            Some(
                PaddedConv2dConfig::new([self.n_channels_out, self.n_channels_out], 3, padding)
                    .with_stride(2)
                    .init(device),
            )
        } else {
            None
        };

        EncoderBlock {
            res1,
            res2,
            downsampler,
        }
    }
}

#[derive(Module, Debug)]
pub struct EncoderBlock<B: Backend> {
    res1: ResnetBlock<B>,
    res2: ResnetBlock<B>,
    downsampler: Option<PaddedConv2d<B>>,
}

impl<B: Backend> EncoderBlock<B> {
    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.res1.forward(x);
        let x = self.res2.forward(x);
        if let Some(d) = self.downsampler.as_ref() {
            d.forward(x)
        } else {
            x
        }
    }
}

#[derive(Config)]
pub struct DecoderBlockConfig {
    n_channels_in: usize,
    n_channels_out: usize,
    upsample: bool,
}

impl DecoderBlockConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> DecoderBlock<B> {
        let res1 = ResnetBlockConfig::new(self.n_channels_in, self.n_channels_out).init(device);
        let res2 = ResnetBlockConfig::new(self.n_channels_out, self.n_channels_out).init(device);
        let res3 = ResnetBlockConfig::new(self.n_channels_out, self.n_channels_out).init(device);
        let upsampler = if self.upsample {
            Some(
                Conv2dConfig::new([self.n_channels_out, self.n_channels_out], [3, 3])
                    .with_padding(PaddingConfig2d::Explicit(1, 1))
                    .init(device),
            )
        } else {
            None
        };

        DecoderBlock {
            res1,
            res2,
            res3,
            upsampler,
        }
    }
}

#[derive(Module, Debug)]
pub struct DecoderBlock<B: Backend> {
    res1: ResnetBlock<B>,
    res2: ResnetBlock<B>,
    res3: ResnetBlock<B>,
    upsampler: Option<Conv2d<B>>,
}

impl<B: Backend> DecoderBlock<B> {
    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.res1.forward(x);
        let x = self.res2.forward(x);
        let x = self.res3.forward(x);

        if let Some(d) = self.upsampler.as_ref() {
            let [n_batch, n_channel, height, width] = x.dims();
            let x = x
                .reshape([n_batch, n_channel, height, 1, width, 1])
                .repeat(&[1, 1, 1, 2, 1, 2])
                .reshape([n_batch, n_channel, 2 * height, 2 * width]);
            d.forward(x)
        } else {
            x
        }
    }
}

#[derive(Config)]
pub struct PaddedConv2dConfig {
    channels: [usize; 2],
    kernel_size: usize,
    #[config(default = 1)]
    stride: usize,
    padding: PaddingCfg,
}

impl PaddedConv2dConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> PaddedConv2d<B> {
        let calc_padding = |p_left, p_right| {
            let n = if p_left >= p_right {
                0
            } else {
                div_roundup(p_right - p_left, self.stride)
            };

            n * self.stride + p_left
        };

        let pad_vertical = calc_padding(self.padding.pad_top, self.padding.pad_bottom);
        let pad_horizontal = calc_padding(self.padding.pad_left, self.padding.pad_right);
        let padding_actual = [pad_vertical, pad_horizontal];

        let conv = Conv2dConfig::new(self.channels, [self.kernel_size, self.kernel_size])
            .with_stride([self.stride, self.stride])
            .with_padding(PaddingConfig2d::Explicit(pad_vertical, pad_horizontal))
            .init(device);

        let kernel_size = self.kernel_size;
        let stride = self.stride;

        let padding = Padding {
            pad_left: self.padding.pad_left, 
            pad_right: self.padding.pad_right, 
            pad_top: self.padding.pad_top, 
            pad_bottom: self.padding.pad_bottom, 
        };

        PaddedConv2d {
            conv,
            kernel_size,
            stride,
            padding,
            padding_actual,
        }
    }
}

fn div_roundup(x: usize, y: usize) -> usize {
    (x + y - 1) / y
}

#[derive(Module, Debug)]
pub struct PaddedConv2d<B: Backend> {
    conv: Conv2d<B>,
    kernel_size: usize,
    stride: usize,
    padding: Padding,
    padding_actual: [usize; 2],
}

impl<B: Backend> PaddedConv2d<B> {
    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let [n_batch, n_channel, height, width] = x.dims();

        let desired_height = (self.padding.pad_top + self.padding.pad_bottom + height
            - self.kernel_size)
            / self.stride
            + 1;
        let desired_width = (self.padding.pad_left + self.padding.pad_right + width
            - self.kernel_size)
            / self.stride
            + 1;

        let skip_vert = (self.padding_actual[0] - self.padding.pad_top) / self.stride;
        let skip_hor = (self.padding_actual[1] - self.padding.pad_left) / self.stride;

        self.conv.forward(x).slice([
            0..n_batch,
            0..n_channel,
            skip_vert..(skip_vert + desired_height),
            skip_hor..(skip_hor + desired_width),
        ])
    }
}

#[derive(Config, Debug)]
pub struct PaddingCfg {
    pad_left: usize,
    pad_right: usize,
    pad_top: usize,
    pad_bottom: usize,
}

#[derive(Module, Clone, Debug)]
pub struct Padding {
    pad_left: usize,
    pad_right: usize,
    pad_top: usize,
    pad_bottom: usize,
}

#[derive(Config)]
pub struct MidConfig {
    n_channel: usize,
}

impl MidConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> Mid<B> {
        let block_1 = ResnetBlockConfig::new(self.n_channel, self.n_channel).init(device);
        let attn = ConvSelfAttentionBlockConfig::new(self.n_channel).init(device);
        let block_2 = ResnetBlockConfig::new(self.n_channel, self.n_channel).init(device);

        Mid {
            block_1,
            attn,
            block_2,
        }
    }
}

#[derive(Module, Debug)]
pub struct Mid<B: Backend> {
    block_1: ResnetBlock<B>,
    attn: ConvSelfAttentionBlock<B>,
    block_2: ResnetBlock<B>,
}

impl<B: Backend> Mid<B> {
    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.block_1.forward(x);
        let x = self.attn.forward(x);
        let x = self.block_2.forward(x);
        x
    }
}

#[derive(Config)]
pub struct ResnetBlockConfig {
    in_channels: usize,
    out_channels: usize,
}

impl ResnetBlockConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> ResnetBlock<B> {
        let norm1 = GroupNormConfig::new(32, self.in_channels).init(device);
        let conv1 = Conv2dConfig::new([self.in_channels, self.out_channels], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);
        let norm2 = GroupNormConfig::new(32, self.out_channels).init(device);
        let conv2 = Conv2dConfig::new([self.out_channels, self.out_channels], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);
        let nin_shortcut = if self.in_channels != self.out_channels {
            Some(Conv2dConfig::new([self.in_channels, self.out_channels], [1, 1]).init(device))
        } else {
            None
        };

        let silu1 = SILU::new();
        let silu2 = SILU::new();

        ResnetBlock {
            norm1,
            silu1,
            conv1,
            norm2,
            silu2,
            conv2,
            nin_shortcut,
        }
    }
}

#[derive(Module, Debug)]
pub struct ResnetBlock<B: Backend> {
    norm1: GroupNorm<B>,
    silu1: SILU,
    conv1: Conv2d<B>,
    norm2: GroupNorm<B>,
    silu2: SILU,
    conv2: Conv2d<B>,
    nin_shortcut: Option<Conv2d<B>>,
}

impl<B: Backend> ResnetBlock<B> {
    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let h = self
            .conv1
            .forward(self.silu1.forward(self.norm1.forward(x.clone())));
        let h = self
            .conv2
            .forward(self.silu2.forward(self.norm2.forward(h)));

        if let Some(ns) = self.nin_shortcut.as_ref() {
            ns.forward(x) + h
        } else {
            x + h
        }
    }
}

#[derive(Config)]
pub struct ConvSelfAttentionBlockConfig {
    n_channel: usize,
}

impl ConvSelfAttentionBlockConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> ConvSelfAttentionBlock<B> {
        let norm = GroupNormConfig::new(32, self.n_channel).init(device);
        let q = Conv2dConfig::new([self.n_channel, self.n_channel], [1, 1]).init(device);
        let k = Conv2dConfig::new([self.n_channel, self.n_channel], [1, 1]).init(device);
        let v = Conv2dConfig::new([self.n_channel, self.n_channel], [1, 1]).init(device);
        let proj_out = Conv2dConfig::new([self.n_channel, self.n_channel], [1, 1]).init(device);

        ConvSelfAttentionBlock {
            norm,
            q,
            k,
            v,
            proj_out,
        }
    }
}

#[derive(Module, Debug)]
pub struct ConvSelfAttentionBlock<B: Backend> {
    norm: GroupNorm<B>,
    q: Conv2d<B>,
    k: Conv2d<B>,
    v: Conv2d<B>,
    proj_out: Conv2d<B>,
}

impl<B: Backend> ConvSelfAttentionBlock<B> {
    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let [n_batch, n_channel, height, width] = x.dims();

        let h = self.norm.forward(x.clone());

        let q = self
            .q
            .forward(h.clone())
            .reshape([n_batch, n_channel, height * width])
            .swap_dims(1, 2);
        let k = self
            .k
            .forward(h.clone())
            .reshape([n_batch, n_channel, height * width])
            .swap_dims(1, 2);
        let v = self
            .v
            .forward(h)
            .reshape([n_batch, n_channel, height * width])
            .swap_dims(1, 2);

        /*let wv = Tensor::from_primitive(B::qkv_attention(
            q.into_primitive(),
            k.into_primitive(),
            v.into_primitive(),
            None,
            1,
        ))
        .swap_dims(1, 2)
        .reshape([n_batch, n_channel, height, width]);*/

        let wv = qkv_attention(
            q,
            k,
            v,
            None,
            1,
        )
        .swap_dims(1, 2)
        .reshape([n_batch, n_channel, height, width]);

        let projected = self.proj_out.forward(wv);

        x + projected
    }
}
