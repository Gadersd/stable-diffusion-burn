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
use crate::model::{load::*, autoencoder::load::load_autoencoder, unet::load::load_unet, clip::load::load_clip};

pub fn load_stable_diffusion<B: Backend>(path: &str, device: &B::Device) -> Result<StableDiffusion<B>, Box<dyn Error>> {
    let n_steps = load_usize::<B>("n_steps", path, device)?;
    let alpha_cumulative_products: Vec<_> = load_tensor::<B, 1>("alphas_cumprod", path, device)?.into_data().value.into_iter()
        .map(|v: <Float as BasicOps<B>>::Elem| v.to_f64().unwrap())
        .collect();
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

