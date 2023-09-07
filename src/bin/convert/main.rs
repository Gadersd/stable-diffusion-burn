use std::env;
use std::error::Error;
use std::process;

use stablediffusion::model::stablediffusion::{load::load_stable_diffusion, StableDiffusion};

use burn::{
    config::Config,
    module::{Module, Param},
    nn,
    tensor::{backend::Backend, Tensor},
};

use burn_ndarray::{NdArrayBackend, NdArrayDevice};

use burn::record::{self, BinFileRecorder, FullPrecisionSettings, Recorder};

fn convert_dump_to_model<B: Backend>(
    dump_path: &str,
    model_name: &str,
    device: &B::Device,
) -> Result<(), Box<dyn Error>> {
    println!("Loading dump...");
    let model: StableDiffusion<B> = load_stable_diffusion(dump_path, device)?;

    println!("Saving model...");
    save_model_file(model, model_name)?;

    Ok(())
}

fn save_model_file<B: Backend>(
    model: StableDiffusion<B>,
    name: &str,
) -> Result<(), record::RecorderError> {
    BinFileRecorder::<FullPrecisionSettings>::new().record(model.into_record(), name.into())
}

fn main() {
    type Backend = NdArrayBackend<f32>;
    let device = NdArrayDevice::Cpu;

    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        eprintln!("Usage: {} <dump_path> <model_name>", args[0]);
        process::exit(1);
    }

    let dump_path = &args[1];
    let model_name = &args[2];

    if let Err(e) = convert_dump_to_model::<Backend>(dump_path, model_name, &device) {
        eprintln!("Failed to convert dump to model: {:?}", e);
        process::exit(1);
    }

    println!("Successfully converted {} to {}", dump_path, model_name);
}
