use std::error::Error;
use burn::tensor::ElementConversion;

use burn::{
    config::Config, 
    module::{Module, Param},
    nn,
    tensor::{
        backend::Backend,
        Tensor,
    },
};

use super::*;
use crate::model::load::*;

pub fn load_mlp<B: Backend>(path: &str, device: &B::Device) -> Result<MLP<B>, Box<dyn Error>> {
    let fc1 = load_linear(&format!("{}/{}", path, "fc1"), device)?;
    let gelu = QuickGELU::new();
    let fc2 = load_linear(&format!("{}/{}", path, "fc2"), device)?;

    let mlp = MLP {
        fc1: fc1,
        gelu: gelu,
        fc2: fc2,
    };

    Ok(mlp)
}

pub fn load_multi_head_self_attention<B: Backend>(path: &str, device: &B::Device) -> Result<MultiHeadSelfAttention<B>, Box<dyn Error>> {
    let n_head = load_usize::<B>("n_head", path, device)?;
    let query = load_linear(&format!("{}/{}", path, "query"), device)?;
    let key = load_linear(&format!("{}/{}", path, "key"), device)?;
    let value = load_linear(&format!("{}/{}", path, "value"), device)?;
    let out = load_linear(&format!("{}/{}", path, "out"), device)?;

    let mhsa = MultiHeadSelfAttention {
        n_head: n_head,
        query: query,
        key: key,
        value: value,
        out: out,
    };

    Ok(mhsa)
}

pub fn load_residual_decoder_attention_block<B: Backend>(path: &str, device: &B::Device) -> Result<ResidualDecoderAttentionBlock<B>, Box<dyn Error>> {
    let mlp = load_mlp(&format!("{}/{}", path, "mlp"), device)?;
    let attn = load_multi_head_self_attention(&format!("{}/{}", path, "attn"), device)?;
    let attn_ln = load_layer_norm(&format!("{}/{}", path, "attn_ln"), device)?;
    let mlp_ln = load_layer_norm(&format!("{}/{}", path, "mlp_ln"), device)?;

    let rdab = ResidualDecoderAttentionBlock {
        attn: attn,
        attn_ln: attn_ln,
        mlp: mlp,
        mlp_ln: mlp_ln,
    };

    Ok(rdab)
}

pub fn load_clip<B: Backend>(path: &str, device: &B::Device) -> Result<CLIP<B>, Box<dyn Error>> {
    let token_embedding = load_embedding(&format!("{}/{}", path, "token_embedding"), device)?;
    let position_embedding = load_tensor("weight", &format!("{}/position_embedding", path), device)?.into();

    let n_layer = load_usize::<B>("n_layer", path, device)?;
    let mut blocks = (0..n_layer)
        .into_iter()
        .map(|i| {
            load_residual_decoder_attention_block::<B>(&format!("{}/blocks/{}", path, i), device)
        }).collect::<Result<Vec<_>, _>>()?;
    
    let layer_norm = load_layer_norm(&format!("{}/{}", path, "layer_norm"), device)?;

    let clip = CLIP {
        token_embedding: token_embedding,
        position_embedding: position_embedding,
        blocks: blocks,
        layer_norm: layer_norm,
    };
    
    Ok(clip)
}
