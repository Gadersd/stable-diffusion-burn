use super::GroupNorm;
use crate::model::load::*;

use std::error::Error;

use burn::{
    config::Config,
    module::{Module, Param},
    nn,
    tensor::{backend::Backend, Tensor},
};

pub fn load_group_norm<B: Backend>(
    path: &str,
    device: &B::Device,
) -> Result<GroupNorm<B>, Box<dyn Error>> {
    let n_group = load_usize::<B>("n_group", path, device)?.into();
    let n_channel = load_usize::<B>("n_channel", path, device)?.into();
    let eps = load_f32::<B>("eps", path, device)?.into();

    let gamma = load_tensor::<B, 1>("weight", path, device)
        .ok()
        .unwrap_or_else(|| Tensor::ones([n_channel], device))
        .into();
    let beta = load_tensor::<B, 1>("bias", path, device)
        .ok()
        .unwrap_or_else(|| Tensor::zeros([n_channel], device))
        .into();

    Ok(GroupNorm {
        n_group,
        n_channel,
        gamma,
        beta,
        eps,
    })
}
