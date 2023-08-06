use std::env;
use std::process;
use std::error::Error;

use stablediffusion::model::stablediffusion::{StableDiffusion, load::load_stable_diffusion};

use burn::{
    config::Config, 
    module::{Module, Param},
    nn,
    tensor::{
        backend::Backend,
        Tensor,
    },
};

cfg_if::cfg_if! {
    if #[cfg(feature = "torch-backend")] {
        use burn_tch::{TchBackend, TchDevice};
    } else if #[cfg(feature = "wgpu-backend")] {
        use burn_wgpu::{WgpuBackend, WgpuDevice, AutoGraphicsApi};
    }
}

use burn::record::{self, Recorder, BinFileRecorder, FullPrecisionSettings};

fn convert_dump_to_model<B: Backend>(dump_path: &str, model_name: &str, device: &B::Device) -> Result<(), Box<dyn Error>> {
    println!("Loading dump...");
    let model: StableDiffusion::<B> = load_stable_diffusion(dump_path, device)?;

    println!("Saving model...");
    save_model_file(model, model_name)?;

    Ok(())
}

fn save_model_file<B: Backend>(model: StableDiffusion<B>, name: &str) -> Result<(), record::RecorderError> {
    BinFileRecorder::<FullPrecisionSettings>::new()
    .record(
        model.into_record(),
        name.into(),
    )
}

fn main() {
    cfg_if::cfg_if! {
        if #[cfg(feature = "torch-backend")] {
            type Backend = TchBackend<f32>;
            let device = TchDevice::Cpu;
        } else if #[cfg(feature = "wgpu-backend")] {
            type Backend = WgpuBackend<AutoGraphicsApi, f32, i32>;
            let device = WgpuDevice::CPU;
        }
    }

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
