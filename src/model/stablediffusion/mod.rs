pub mod load;

use burn::{
    config::Config,
    module::{Module, Param},
    tensor::{backend::Backend, BasicOps, Data, Distribution, Float, Int, Tensor},
    tensor::cast::ToElement, 
};

use num_traits::ToPrimitive;

//use crate::backend::Backend as MyBackend;

use super::autoencoder::{Autoencoder, AutoencoderConfig};
use super::clip::{CLIPConfig, CLIP};
use super::unet::{UNet, UNetConfig};
use crate::tokenizer::SimpleTokenizer;

#[derive(Config)]
pub struct StableDiffusionConfig {}

impl StableDiffusionConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> StableDiffusion<B> {
        let n_steps = 1000;
        let alpha_cumulative_products = Param::from_tensor(offset_cosine_schedule_cumprod::<B>(n_steps as i64, device));

        let autoencoder = AutoencoderConfig::new().init(device);
        let diffusion = UNetConfig::new().init(device);
        let clip = CLIPConfig::new(49408, 768, 12, 77, 12).init(device);

        StableDiffusion {
            n_steps,
            alpha_cumulative_products,
            autoencoder,
            diffusion,
            clip,
        }
    }
}

#[derive(Module, Debug)]
pub struct StableDiffusion<B: Backend> {
    n_steps: usize,
    alpha_cumulative_products: Param<Tensor<B, 1>>,
    autoencoder: Autoencoder<B>,
    diffusion: UNet<B>,
    clip: CLIP<B>,
}

impl<B: Backend> StableDiffusion<B> {
    pub fn sample_image(
        &self,
        context: Tensor<B, 3>,
        unconditional_context: Tensor<B, 2>,
        unconditional_guidance_scale: f64,
        n_steps: usize,
    ) -> Vec<Vec<u8>> {
        let [n_batch, _, _] = context.dims();

        let latent = self.sample_latent(
            context,
            unconditional_context,
            unconditional_guidance_scale,
            n_steps,
        );
        self.latent_to_image(latent)
    }

    pub fn latent_to_image(&self, latent: Tensor<B, 4>) -> Vec<Vec<u8>> {
        let [n_batch, _, _, _] = latent.dims();
        let image = self.autoencoder.decode_latent(latent * (1.0 / 0.18215));

        let n_channel = 3;
        let height = 512;
        let width = 512;
        let num_elements_per_image = n_channel * height * width;

        // correct size and scale and reorder to
        let image = (image + 1.0) / 2.0;
        let image = image
            .reshape([n_batch, n_channel, height, width])
            .swap_dims(1, 2)
            .swap_dims(2, 3)
            .mul_scalar(255.0);

        let flattened: Vec<B::FloatElem> = image.into_data().to_vec().unwrap();

        (0..n_batch)
            .into_iter()
            .map(|b| {
                let start = b * num_elements_per_image;
                let end = start + num_elements_per_image;

                flattened[start..end]
                    .into_iter()
                    .map(|v| v.to_f64().min(255.0).max(0.0) as u8)
                    .collect()
            })
            .collect()
    }

    pub fn sample_latent(
        &self,
        context: Tensor<B, 3>,
        unconditional_context: Tensor<B, 2>,
        unconditional_guidance_scale: f64,
        n_steps: usize,
    ) -> Tensor<B, 4> {
        let device = context.device();

        let step_size = self.n_steps / n_steps;

        let [n_batches, _, _] = context.dims();

        let gen_noise = || {
            Tensor::random([n_batches, 4, 64, 64], Distribution::Normal(0.0, 1.0), &device)
        };

        let sigma = 0.0; // Use deterministic diffusion

        let mut latent = gen_noise();

        for t in (0..self.n_steps).rev().step_by(step_size) {
            let current_alpha: f64 = self
                .alpha_cumulative_products
                .val()
                .slice([t..t + 1])
                .into_scalar()
                .to_f64();

            let prev_alpha: f64 = if t >= step_size {
                let i = t - step_size;
                self.alpha_cumulative_products
                    .val()
                    .slice([i..i + 1])
                    .into_scalar()
                    .to_f64()
            } else {
                1.0
            };

            let sqrt_noise = (1.0 - current_alpha).sqrt();

            let timestep = Tensor::from_ints([t as i32], &device);
            let pred_noise = self.forward_diffuser(
                latent.clone(),
                timestep,
                context.clone(),
                unconditional_context.clone(),
                unconditional_guidance_scale,
            );
            let predx0 = (latent - pred_noise.clone() * sqrt_noise) / current_alpha.sqrt();
            let dir_latent = pred_noise * (1.0 - prev_alpha - sigma * sigma).sqrt();

            let prev_latent = predx0 * prev_alpha.sqrt() + dir_latent + gen_noise() * sigma;
            latent = prev_latent;
        }

        latent
    }

    fn forward_diffuser(
        &self,
        latent: Tensor<B, 4>,
        timestep: Tensor<B, 1, Int>,
        context: Tensor<B, 3>,
        unconditional_context: Tensor<B, 2>,
        unconditional_guidance_scale: f64,
    ) -> Tensor<B, 4> {
        let [n_batch, _, _, _] = latent.dims();
        //let latent = latent.repeat(0, 2);

        let unconditional_latent = self.diffusion.forward(
            latent.clone(),
            timestep.clone(),
            unconditional_context.unsqueeze().repeat(&[0, n_batch]),
        );

        let conditional_latent = self.diffusion.forward(latent, timestep, context);

        /*let latent = self.diffusion.forward(
            latent.repeat(0, 2),
            timestep.repeat(0, 2),
            Tensor::cat(vec![unconditional_context.unsqueeze::<3>(), context], 0)
        );

        let unconditional_latent = latent.clone().slice([0..n_batch]);
        let conditional_latent = latent.slice([n_batch..2 * n_batch]);*/

        unconditional_latent.clone()
            + (conditional_latent - unconditional_latent) * unconditional_guidance_scale
    }

    pub fn unconditional_context(&self, tokenizer: &SimpleTokenizer) -> Tensor<B, 2> {
        self.context(tokenizer, "").squeeze(0)
    }

    pub fn context(&self, tokenizer: &SimpleTokenizer, text: &str) -> Tensor<B, 3> {
        let device = &self.clip.devices()[0];
        let text = format!("<|startoftext|>{}<|endoftext|>", text);
        let tokenized: Vec<_> = tokenizer
            .encode(&text)
            .into_iter()
            .map(|v| v as i32)
            .collect();

        self.clip.forward(
            Tensor::<B, 1, Int>::from_ints(&tokenized[..], device)
                .unsqueeze(),
        )
    }
}

use std::f64::consts::PI;

fn cosine_schedule<B: Backend>(n_steps: i64, device: &B::Device) -> Tensor<B, 1> {
    Tensor::arange(1..n_steps + 1, device)
        .float()
        .mul_scalar(PI * 0.5 / n_steps as f64)
        .cos()
}

fn offset_cosine_schedule<B: Backend>(n_steps: i64, device: &B::Device) -> Tensor<B, 1> {
    let min_signal_rate: f64 = 0.02;
    let max_signal_rate: f64 = 0.95;
    let start_angle = max_signal_rate.acos();
    let end_angle = min_signal_rate.acos();

    let times = Tensor::arange(1..n_steps + 1, device).float();

    let diffusion_angles = times * ((end_angle - start_angle) / n_steps as f64) + start_angle;
    diffusion_angles.cos()
}

fn offset_cosine_schedule_cumprod<B: Backend>(n_steps: i64, device: &B::Device) -> Tensor<B, 1> {
    offset_cosine_schedule::<B>(n_steps, device).powf_scalar(2.0)
}
