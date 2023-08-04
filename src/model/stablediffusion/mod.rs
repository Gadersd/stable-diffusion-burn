pub mod load;

use burn::{
    config::Config, 
    module::Module,
    tensor::{
        backend::Backend,
        Tensor,
        Int, 
        Float, 
        BasicOps, 
        Data, 
        Distribution, 
    },
};

use num_traits::ToPrimitive;

use super::autoencoder::{Autoencoder, AutoencoderConfig};
use super::unet::{UNet, UNetConfig};
use super::clip::{CLIP, CLIPConfig};
use crate::tokenizer::SimpleTokenizer;

#[derive(Config)]
pub struct StableDiffusionConfig {

}

impl StableDiffusionConfig {
    fn init<B: Backend>(&self) -> StableDiffusion<B> {
        let n_steps = 1000;
        let alpha_cumulative_products = offset_cosine_schedule_cumprod::<B>(n_steps)
            .into_data().value
            .into_iter()
            .map(|v: <Float as BasicOps<B>>::Elem| v.to_f64().unwrap()).collect();

        let autoencoder = AutoencoderConfig::new().init();
        let diffusion = UNetConfig::new().init();
        let clip = CLIPConfig::new(49408, 768, 12, 77, 12).init();

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
    alpha_cumulative_products: Vec<f64>, 
    autoencoder: Autoencoder<B>, 
    diffusion: UNet<B>, 
    clip: CLIP<B>, 
}

impl<B: Backend> StableDiffusion<B> {
    pub fn sample_image(&self, context: Tensor<B, 3>, unconditional_context: Tensor<B, 2>, unconditional_guidance_scale: f64, n_steps: usize) -> Vec<Vec<u8>> {
        let [n_batch, _, _] = context.dims();

        let latent = self.sample_latent(context, unconditional_context, unconditional_guidance_scale, n_steps);
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

        let flattened: Vec<_> = image.
            into_data().
            value;

        (0..n_batch).into_iter().map(|b| {
            let start = b * num_elements_per_image;
            let end = start + num_elements_per_image;

            flattened[start..end].into_iter().map(|v| v.to_u8().unwrap()).collect()
        }).collect()
    }

    pub fn sample_latent(&self, context: Tensor<B, 3>, unconditional_context: Tensor<B, 2>, unconditional_guidance_scale: f64, n_steps: usize) -> Tensor<B, 4> {
        assert!(self.n_steps % n_steps == 0);

        let step_size = self.n_steps / n_steps;

        let [n_batches, _, _] = context.dims();

        let gen_noise = || {
            Tensor::random([n_batches, 4, 64, 64], Distribution::Normal(0.0, 1.0) )
        };

        let sigma = 0.0; // Use deterministic diffusion

        let mut latent = gen_noise();

        for t in (0..self.n_steps).rev().step_by(step_size) {
            let current_alpha = self.alpha_cumulative_products[t];
            let prev_alpha = if t >= step_size {
                self.alpha_cumulative_products[t - step_size]
            } else {
                1.0
            };

            let sqrt_noise = (1.0 - current_alpha).sqrt();

            let timestep = Tensor::from_ints([t as i32]);
            let pred_noise = self.forward_diffuser(latent.clone(), timestep, context.clone(), unconditional_context.clone(), unconditional_guidance_scale);

            let predx0 = (latent - pred_noise.clone() * sqrt_noise) / current_alpha.sqrt();
            let dir_latent = pred_noise * (1.0 - prev_alpha - sigma * sigma).sqrt();

            let prev_latent = predx0 * prev_alpha.sqrt() + dir_latent + gen_noise() * sigma;
            latent = prev_latent;
        }

        latent
    }

    fn forward_diffuser(&self, latent: Tensor<B, 4>, timestep: Tensor<B, 1, Int>, context: Tensor<B, 3>, unconditional_context: Tensor<B, 2>, unconditional_guidance_scale: f64) -> Tensor<B, 4> {
        let [n_batch, n_channel, height, width] = latent.dims();
        let latent = latent.repeat(0, 2);

        let latent = self.diffusion.forward(
            latent.repeat(0, 2), 
            timestep.repeat(0, 2), 
            Tensor::cat(vec![unconditional_context.unsqueeze::<3>(), context], 0)
        );

        let unconditional_latent = latent.clone().slice([0..n_batch]);
        let conditional_latent = latent.slice([n_batch..2 * n_batch]);

        unconditional_latent.clone() + (conditional_latent - unconditional_latent) * unconditional_guidance_scale
    }

    pub fn unconditional_context(&self, tokenizer: &SimpleTokenizer) -> Tensor<B, 2> {
        self.context(tokenizer, "").squeeze(0)
    }

    pub fn context(&self, tokenizer: &SimpleTokenizer, text: &str) -> Tensor<B, 3> {
        let text = format!("<|startoftext|>{}<|endoftext|>", text);
        let tokenized: Vec<_> = tokenizer.encode(&text).into_iter().map(|v| v as i32).collect();

        self.clip.forward(Tensor::from_ints(&tokenized[..]).unsqueeze())
    }
}

use crate::helper::to_float;
use std::f64::consts::PI;

fn cosine_schedule<B: Backend>(n_steps: usize) -> Tensor<B, 1> {
    to_float(Tensor::arange(1..n_steps + 1))
        .mul_scalar(PI * 0.5 / n_steps as f64)
        .cos()
}

fn offset_cosine_schedule<B: Backend>(n_steps: usize) -> Tensor<B, 1> {
    let min_signal_rate: f64 = 0.02;
    let max_signal_rate: f64 = 0.95;
    let start_angle = max_signal_rate.acos();
    let end_angle = min_signal_rate.acos();

    let times = Tensor::arange(1..n_steps + 1);

    let diffusion_angles = to_float(times) * ( (end_angle - start_angle) / n_steps as f64) + start_angle;
    diffusion_angles.cos()
}

fn offset_cosine_schedule_cumprod<B: Backend>(n_steps: usize) -> Tensor<B, 1> {
    offset_cosine_schedule::<B>(n_steps).powf(2.0)
}