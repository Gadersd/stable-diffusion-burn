use burn::tensor::ElementConversion;
use std::error::Error;

use burn::{
    config::Config,
    module::{Module, Param},
    nn,
    tensor::{backend::Backend, Tensor},
};

use super::*;
use crate::model::{
    autoencoder::load::load_autoencoder, clip::load::load_clip, load::*, unet::load::load_unet,
};

pub fn load_stable_diffusion<B: Backend>(
    path: &str,
    device: &B::Device,
) -> Result<StableDiffusion<B>, Box<dyn Error>> {
    let n_steps = load_usize::<B>("n_steps", path, device)?;
    let alpha_cumulative_products = Param::from_tensor(load_tensor::<B, 1>("alphas_cumprod", path, device)?);
    let autoencoder = load_autoencoder(&format!("{}/{}", path, "autoencoder"), device)?;
    let diffusion = load_unet(&format!("{}/{}", path, "unet"), device)?;
    let clip = load_clip(&format!("{}/{}", path, "clip"), device)?;

    Ok(StableDiffusion {
        n_steps,
        alpha_cumulative_products,
        autoencoder,
        diffusion,
        clip,
    })
}
