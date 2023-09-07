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

pub fn load_res_block<B: Backend>(
    path: &str,
    device: &B::Device,
) -> Result<ResBlock<B>, Box<dyn Error>> {
    let norm_in = load_group_norm::<B>(&format!("{}/{}", path, "norm_in"), device)?;
    let conv_in = load_conv2d::<B>(&format!("{}/{}", path, "conv_in"), device)?;
    let lin_embed = load_linear::<B>(&format!("{}/{}", path, "lin_embed"), device)?;
    let norm_out = load_group_norm::<B>(&format!("{}/{}", path, "norm_out"), device)?;
    let conv_out = load_conv2d::<B>(&format!("{}/{}", path, "conv_out"), device)?;
    let skip_connection = load_conv2d::<B>(&format!("{}/{}", path, "skip_connection"), device).ok();

    let res_block = ResBlock {
        norm_in: norm_in,
        silu_in: SILU::new(),
        conv_in: conv_in,
        silu_embed: SILU::new(),
        lin_embed: lin_embed,
        norm_out: norm_out,
        silu_out: SILU::new(),
        conv_out: conv_out,
        skip_connection: skip_connection,
    };

    Ok(res_block)
}

pub fn load_multi_head_attention<B: Backend>(
    path: &str,
    device: &B::Device,
) -> Result<MultiHeadAttention<B>, Box<dyn Error>> {
    let n_head = load_usize::<B>("n_head", path, device)?;
    let query = load_linear::<B>(&format!("{}/{}", path, "query"), device)?;
    let key = load_linear::<B>(&format!("{}/{}", path, "key"), device)?;
    let value = load_linear::<B>(&format!("{}/{}", path, "value"), device)?;
    let out = load_linear::<B>(&format!("{}/{}", path, "out"), device)?;

    let multi_head_attention = MultiHeadAttention {
        n_head: n_head,
        query: query,
        key: key,
        value: value,
        out: out,
    };

    Ok(multi_head_attention)
}

pub fn load_geglu<B: Backend>(path: &str, device: &B::Device) -> Result<GEGLU<B>, Box<dyn Error>> {
    let proj = load_linear::<B>(&format!("{}/{}", path, "proj"), device)?;

    let geglue = GEGLU {
        proj: proj,
        gelu: GELU::new(), // Assuming GELU::new() initializes a new GELU struct
    };

    Ok(geglue)
}

pub fn load_mlp<B: Backend>(path: &str, device: &B::Device) -> Result<MLP<B>, Box<dyn Error>> {
    let geglu = load_geglu::<B>(&format!("{}/{}", path, "geglu"), device)?;
    let lin = load_linear::<B>(&format!("{}/{}", path, "lin"), device)?;

    let mlp = MLP {
        geglu: geglu,
        lin: lin,
    };

    Ok(mlp)
}

pub fn load_transformer_block<B: Backend>(
    path: &str,
    device: &B::Device,
) -> Result<TransformerBlock<B>, Box<dyn Error>> {
    let norm1 = load_layer_norm::<B>(&format!("{}/{}", path, "norm1"), device)?;
    let attn1 = load_multi_head_attention::<B>(&format!("{}/{}", path, "attn1"), device)?;
    let norm2 = load_layer_norm::<B>(&format!("{}/{}", path, "norm2"), device)?;
    let attn2 = load_multi_head_attention::<B>(&format!("{}/{}", path, "attn2"), device)?;
    let norm3 = load_layer_norm::<B>(&format!("{}/{}", path, "norm3"), device)?;
    let mlp = load_mlp::<B>(&format!("{}/{}", path, "mlp"), device)?;

    let transformer_block = TransformerBlock {
        norm1: norm1,
        attn1: attn1,
        norm2: norm2,
        attn2: attn2,
        norm3: norm3,
        mlp: mlp,
    };

    Ok(transformer_block)
}

pub fn load_spatial_transformer<B: Backend>(
    path: &str,
    device: &B::Device,
) -> Result<SpatialTransformer<B>, Box<dyn Error>> {
    let norm = load_group_norm::<B>(&format!("{}/{}", path, "norm"), device)?;
    let proj_in = load_conv2d::<B>(&format!("{}/{}", path, "proj_in"), device)?;
    let transformer = load_transformer_block::<B>(&format!("{}/{}", path, "transformer"), device)?;
    let proj_out = load_conv2d::<B>(&format!("{}/{}", path, "proj_out"), device)?;

    let spatial_transformer = SpatialTransformer {
        norm: norm,
        proj_in: proj_in,
        transformer: transformer,
        proj_out: proj_out,
    };

    Ok(spatial_transformer)
}

pub fn load_upsample<B: Backend>(
    path: &str,
    device: &B::Device,
) -> Result<Upsample<B>, Box<dyn Error>> {
    let conv = load_conv2d::<B>(&format!("{}/{}", path, "conv"), device)?;

    let upsample = Upsample { conv: conv };

    Ok(upsample)
}

pub fn load_downsample<B: Backend>(
    path: &str,
    device: &B::Device,
) -> Result<Downsample<B>, Box<dyn Error>> {
    load_conv2d(path, device)
}

pub fn load_res_transformer_res<B: Backend>(
    path: &str,
    device: &B::Device,
) -> Result<ResTransformerRes<B>, Box<dyn Error>> {
    let res1 = load_res_block::<B>(&format!("{}/{}", path, "res1"), device)?; // Assuming load_res_block function
    let transformer =
        load_spatial_transformer::<B>(&format!("{}/{}", path, "transformer"), device)?;
    let res2 = load_res_block::<B>(&format!("{}/{}", path, "res2"), device)?;

    let res_transformer_res = ResTransformerRes {
        res1: res1,
        transformer: transformer,
        res2: res2,
    };

    Ok(res_transformer_res)
}

pub fn load_res_transformer_upsample<B: Backend>(
    path: &str,
    device: &B::Device,
) -> Result<ResTransformerUpsample<B>, Box<dyn Error>> {
    let res = load_res_block::<B>(&format!("{}/{}", path, "res"), device)?;
    let transformer =
        load_spatial_transformer::<B>(&format!("{}/{}", path, "transformer"), device)?;
    let upsample = load_upsample::<B>(&format!("{}/{}", path, "upsample"), device)?;

    let res_transformer_upsample = ResTransformerUpsample {
        res: res,
        transformer: transformer,
        upsample: upsample,
    };

    Ok(res_transformer_upsample)
}

pub fn load_res_upsample<B: Backend>(
    path: &str,
    device: &B::Device,
) -> Result<ResUpSample<B>, Box<dyn Error>> {
    let res = load_res_block::<B>(&format!("{}/{}", path, "res"), device)?;
    let upsample = load_upsample::<B>(&format!("{}/{}", path, "upsample"), device)?;

    let res_upsample = ResUpSample {
        res: res,
        upsample: upsample,
    };

    Ok(res_upsample)
}

pub fn load_res_transformer<B: Backend>(
    path: &str,
    device: &B::Device,
) -> Result<ResTransformer<B>, Box<dyn Error>> {
    let res = load_res_block::<B>(&format!("{}/{}", path, "res"), device)?;
    let transformer =
        load_spatial_transformer::<B>(&format!("{}/{}", path, "transformer"), device)?;

    let res_transformer = ResTransformer {
        res: res,
        transformer: transformer,
    };

    Ok(res_transformer)
}

pub fn load_unet_input_blocks<B: Backend>(
    path: &str,
    device: &B::Device,
) -> Result<UNetInputBlocks<B>, Box<dyn Error>> {
    let conv = load_conv2d::<B>(&format!("{}/{}", path, "conv"), device)?;
    let rt1 = load_res_transformer::<B>(&format!("{}/{}", path, "rt1"), device)?;
    let rt2 = load_res_transformer::<B>(&format!("{}/{}", path, "rt2"), device)?;
    let d1 = load_downsample::<B>(&format!("{}/{}", path, "d1"), device)?;
    let rt3 = load_res_transformer::<B>(&format!("{}/{}", path, "rt3"), device)?;
    let rt4 = load_res_transformer::<B>(&format!("{}/{}", path, "rt4"), device)?;
    let d2 = load_downsample::<B>(&format!("{}/{}", path, "d2"), device)?;
    let rt5 = load_res_transformer::<B>(&format!("{}/{}", path, "rt5"), device)?;
    let rt6 = load_res_transformer::<B>(&format!("{}/{}", path, "rt6"), device)?;
    let d3 = load_downsample::<B>(&format!("{}/{}", path, "d3"), device)?;
    let r1 = load_res_block::<B>(&format!("{}/{}", path, "r1"), device)?;
    let r2 = load_res_block::<B>(&format!("{}/{}", path, "r2"), device)?;

    let unet_input_blocks = UNetInputBlocks {
        conv: conv,
        rt1: rt1,
        rt2: rt2,
        d1: d1,
        rt3: rt3,
        rt4: rt4,
        d2: d2,
        rt5: rt5,
        rt6: rt6,
        d3: d3,
        r1: r1,
        r2: r2,
    };

    Ok(unet_input_blocks)
}

pub fn load_unet_output_blocks<B: Backend>(
    path: &str,
    device: &B::Device,
) -> Result<UNetOutputBlocks<B>, Box<dyn Error>> {
    let r1 = load_res_block::<B>(&format!("{}/{}", path, "r1"), device)?;
    let r2 = load_res_block::<B>(&format!("{}/{}", path, "r2"), device)?;
    let ru = load_res_upsample::<B>(&format!("{}/{}", path, "ru"), device)?;
    let rt1 = load_res_transformer::<B>(&format!("{}/{}", path, "rt1"), device)?;
    let rt2 = load_res_transformer::<B>(&format!("{}/{}", path, "rt2"), device)?;
    let rtu1 = load_res_transformer_upsample::<B>(&format!("{}/{}", path, "rtu1"), device)?;
    let rt3 = load_res_transformer::<B>(&format!("{}/{}", path, "rt3"), device)?;
    let rt4 = load_res_transformer::<B>(&format!("{}/{}", path, "rt4"), device)?;
    let rtu2 = load_res_transformer_upsample::<B>(&format!("{}/{}", path, "rtu2"), device)?;
    let rt5 = load_res_transformer::<B>(&format!("{}/{}", path, "rt5"), device)?;
    let rt6 = load_res_transformer::<B>(&format!("{}/{}", path, "rt6"), device)?;
    let rt7 = load_res_transformer::<B>(&format!("{}/{}", path, "rt7"), device)?;

    Ok(UNetOutputBlocks {
        r1,
        r2,
        ru,
        rt1,
        rt2,
        rtu1,
        rt3,
        rt4,
        rtu2,
        rt5,
        rt6,
        rt7,
    })
}

pub fn load_unet<B: Backend>(path: &str, device: &B::Device) -> Result<UNet<B>, Box<dyn Error>> {
    let lin1_time_embed = load_linear::<B>(&format!("{}/{}", path, "lin1_time_embed"), device)?;
    let silu_time_embed = SILU::new(); // Assuming SILU::new() initializes a new SILU struct
    let lin2_time_embed = load_linear::<B>(&format!("{}/{}", path, "lin2_time_embed"), device)?;
    let input_blocks =
        load_unet_input_blocks::<B>(&format!("{}/{}", path, "input_blocks"), device)?;
    let middle_block =
        load_res_transformer_res::<B>(&format!("{}/{}", path, "middle_block"), device)?;
    let output_blocks =
        load_unet_output_blocks::<B>(&format!("{}/{}", path, "output_blocks"), device)?;
    let norm_out = load_group_norm::<B>(&format!("{}/{}", path, "norm_out"), device)?;
    let silu_out = SILU::new(); // Assuming SILU::new() initializes a new SILU struct
    let conv_out = load_conv2d::<B>(&format!("{}/{}", path, "conv_out"), device)?;

    Ok(UNet {
        lin1_time_embed,
        silu_time_embed,
        lin2_time_embed,
        input_blocks,
        middle_block,
        output_blocks,
        norm_out,
        silu_out,
        conv_out,
    })
}
