use stablediffusion::{tokenizer::SimpleTokenizer, model::stablediffusion::{*, load::load_stable_diffusion}};

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

use std::env;
use std::io;
use std::process;

use burn::record::{self, Recorder, BinFileRecorder, FullPrecisionSettings};

fn load_stable_diffusion_model_file<B: Backend>(filename: &str) -> Result<StableDiffusion<B>, record::RecorderError> {
    BinFileRecorder::<FullPrecisionSettings>::new()
    .load(filename.into())
    .map(|record| StableDiffusionConfig::new().init().load_record(record))
}

fn main() {
    type Backend = TchBackend<f32>;
    //let device = TchDevice::Cpu;
    let device = TchDevice::Cuda(0);

    let args: Vec<String> = std::env::args().collect();
    if args.len() != 7 {
        eprintln!("Usage: {} <model_type(burn or dump)> <model_name> <unconditional_guidance_scale> <n_diffusion_steps> <prompt> <output_image_name>", args[0]);
        process::exit(1);
    }

    let model_type = &args[1];
    let model_name = &args[2];
    let unconditional_guidance_scale: f64 = args[3].parse().unwrap_or_else(|_| {
        eprintln!("Error: Invalid unconditional guidance scale.");
        process::exit(1);
    });
    let n_steps: usize = args[4].parse().unwrap_or_else(|_| {
        eprintln!("Error: Invalid number of diffusion steps.");
        process::exit(1);
    });
    let prompt = &args[5];
    let output_image_name = &args[6];

    println!("Loading tokenizer...");
    let tokenizer = SimpleTokenizer::new().unwrap();
    println!("Loading model...");
    let sd: StableDiffusion<Backend> = if model_type == "burn" {
        load_stable_diffusion_model_file(model_name).unwrap_or_else(|err| {
            eprintln!("Error loading model: {}", err);
            process::exit(1);
        })
    } else {
        load_stable_diffusion(model_name, &device).unwrap_or_else(|err| {
            eprintln!("Error loading model dump: {}", err);
            process::exit(1);
        })
    };
    
     
    let sd = sd.to_device(&device);

    let unconditional_context = sd.unconditional_context(&tokenizer);
    let context = sd.context(&tokenizer, prompt).unsqueeze();

    println!("Sampling image...");
    let images = sd.sample_image(context, unconditional_context, unconditional_guidance_scale, n_steps);
    save_images(&images, output_image_name, 512, 512).unwrap_or_else(|err| {
        eprintln!("Error saving image: {}", err);
        process::exit(1);
    });
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