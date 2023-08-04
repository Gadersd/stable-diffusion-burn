use stablediffusion::{tokenizer::SimpleTokenizer, model::clip::{*, load::*}, 
model::autoencoder::{*, load::*}, 
model::groupnorm::*, 
model::unet::{*, load::*}, 
model::stablediffusion::{*, load::*}};

use burn::{
    config::Config, 
    module::{Module, Param},
    nn,
    tensor::{
        backend::Backend,
        Tensor,
    },
};
use burn_tch::{TchBackend, TchDevice};

fn print_tensor<B: Backend>(x: Tensor<B, 4>) {
    let data = x/*.slice([0..1, 0..4, 0..10])*/.into_data();
    println!("{:?}", data);
}

use stablediffusion::helper::to_float;

fn main() {
    type Backend = TchBackend<f32>;
    //let device = TchDevice::Cpu;
    let device = TchDevice::Cuda(0);

    /*let norm: nn::LayerNorm<Backend> = nn::LayerNormConfig::new(3).init();
    let tensor = Tensor::from_floats([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape([2, 3]);

    let out = norm.forward(tensor);

    println!("{:?}", out.into_data());

    return;*/

    /*let n_channel = 6;
    let norm: nn::LayerNorm<Backend> = nn::LayerNormConfig::new(10).init();
    let height = 10;
    let width = 10;
    let n_elements = height * width * n_channel;
    let t: Tensor<Backend, 4> = to_float(Tensor::arange(0..n_elements)).mul_scalar(10.0 / n_elements as f64).sin().reshape([1, n_channel, height, width]); 
    let out = layernorm(t, 1e-5); //norm.forward(t);
    println!("{:?}", out.to_data());
    return;*/

    /*let clip: CLIP<Backend> = load_clip("params", &device).unwrap();
    let input = Tensor::from_ints([3, 1]);

    let output = clip.forward(input.unsqueeze());
    print_tensor(output);*/

    /*let autoencoder: Autoencoder<Backend> = load_autoencoder("params", &device).unwrap();
    let input = Tensor::zeros([1, 3, 10, 10]);
    let output = autoencoder.forward(input);
    print_tensor(output);*/

    /*let unet: UNet<Backend> = load_unet("params", &device).unwrap();
    let input = Tensor::zeros([1, 4, 64, 64]);
    let context = Tensor::from_floats([0.5, 1.3]).repeat(0, 768 / 2).unsqueeze();
    let timesteps = Tensor::from_floats([1.0]);

    let output = unet.forward(input, timesteps, context);*/
    //print_tensor(output);

    println!("Loading tokenizer...");
    let tokenizer = SimpleTokenizer::new().unwrap();

    println!("Loading Stable Diffusion...");
    let sd: StableDiffusion<Backend> = load_stable_diffusion("params", &device).unwrap();
    let sd = sd.to_device(&device);

    let unconditional_guidance_scale = 7.5;
    let unconditional_context = sd.unconditional_context(&tokenizer);
    let context = sd.context(&tokenizer, "A wine glass filled with pink flower petals.").unsqueeze();

    let n_steps = 100;

    println!("Sampling images...");
    let images = sd.sample_image(context, unconditional_context, unconditional_guidance_scale, n_steps);

    println!("Saving images...");
    save_images(&images, "image_samples/", 512, 512).unwrap();
}

use image::{self, ImageResult, ColorType::Rgb8};

fn save_images(images: &Vec<Vec<u8>>, basepath: &str, width: u32, height: u32) -> ImageResult<()> {
    for (index, img_data) in images.iter().enumerate() {
        let path = format!("{}{}.png", basepath, index);
        image::save_buffer(path, &img_data[..], width, height, Rgb8)?;
    }

    Ok(())
}

// save red test image
fn save_test_image() -> ImageResult<()> {
    let width = 256;
    let height = 256;
    let raw: Vec<_> = (0..width * height).into_iter().flat_map(|i| {
        let row = i / width;
        let red = (255.0 * row as f64 / height as f64) as u8;

        [red, 0, 0]
    }).collect();

    image::save_buffer("red.png", &raw[..], width, height, Rgb8)
}