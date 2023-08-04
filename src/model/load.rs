use std::error::Error;
use std::io::Read;
use npy::{self, NpyData};
use num_traits::cast::ToPrimitive;

use burn::{
    config::Config, 
    module::{Module, Param},
    nn::{self, conv},
    tensor::{
        backend::Backend,
        Tensor,
        Data, 
    },
};

use burn::tensor::ElementConversion;

pub fn numpy_to_tensor<B: Backend, const D: usize>(numpy_data: NpyData<f32>, device: &B::Device) -> Tensor<B, D> {
    let mut v = numpy_data.to_vec();

    let shape: Vec<_> = v[0..D].into_iter().map(|&v| v as usize).collect();
    let data: Vec<B::FloatElem> = v[D..].into_iter().map(|e| e.elem()).collect();
    
    Tensor::from_data_device(Data::new(data, shape.into()), device)
}

pub fn load_tensor<B: Backend, const D: usize>(name: &str, path: &str, device: &B::Device) -> Result<Tensor<B, D>, Box<dyn Error>> {
    let tensor_path = format!("{}/{}.npy", path, name);

    let mut buf = vec![];
    std::fs::File::open(&tensor_path)?
        .read_to_end(&mut buf)?;

    let tensor_numpy: NpyData<f32> = NpyData::from_bytes(&buf)?;

    let tensor = numpy_to_tensor(tensor_numpy, device);

    println!("{}", tensor_path);

    Ok(tensor)
}

pub fn load_f32<B: Backend>(name: &str, path: &str, device: &B::Device) -> Result<f32, Box<dyn Error>> {
    load_tensor::<B, 1>(name, path, device).map(|t| t.into_scalar().to_f32().unwrap())
}

pub fn load_usize<B: Backend>(name: &str, path: &str, device: &B::Device) -> Result<usize, Box<dyn Error>> {
    load_tensor::<B, 1>(name, path, device).map(|t| t.into_scalar().to_usize().unwrap())
}

pub fn load_linear<B: Backend>(path: &str, device: &B::Device) -> Result<nn::Linear<B>, Box<dyn Error>> {
    let weight = load_tensor::<B, 2>("weight", path, device)?;
    let bias = load_tensor::<B, 1>("bias", path, device).ok();

    let record = nn::LinearRecord {
        weight: weight.into(),
        bias: bias.map(|t| t.into()),
    };

    let linear: nn::Linear<B> = nn::LinearConfig::new(3, 3).init_with(record);
    Ok(linear)
}

pub fn load_embedding<B: Backend>(path: &str, device: &B::Device) -> Result<nn::Embedding<B>, Box<dyn Error>> {
    let weight = load_tensor::<B, 2>("weight", path, device)?;
    let [n_vocab, n_state] = weight.dims();

    let record = nn::EmbeddingRecord {
        weight: weight.into(),
    };

    let embedding = nn::EmbeddingConfig::new(n_vocab, n_state).init_with(record);
    Ok(embedding)
}

pub fn load_layer_norm<B: Backend>(path: &str, device: &B::Device) -> Result<nn::LayerNorm<B>, Box<dyn Error>> {
    let weight = load_tensor::<B, 1>("weight", path, device)?;
    let bias = load_tensor::<B, 1>("bias", path, device)?;
    let eps = load_f32::<B>("eps", path, device)? as f64;

    let [n_state] = weight.dims();

    let record = nn::LayerNormRecord {
        gamma: weight.into(),
        beta: bias.into(),
        epsilon: <f64 as Module<B>>::into_record(eps), 
    };

    let layer_norm: nn::LayerNorm<B> = nn::LayerNormConfig::new(n_state).init_with(record);

    Ok(layer_norm)
}


/*pub fn load_rmsnorm<B: Backend>(path: &str, device: &B::Device) -> Result<RMSNorm<B>, Box<dyn Error>> {
    let weight = load_tensor::<B, 1>("weight", path, device)?;
    let eps = load_f32::<B>("eps", path, device)?.into();

    let rmsnorm =  RMSNorm { 
        weight: weight.into(), 
        eps: eps
    };
    
    Ok(rmsnorm)
}*/

pub fn load_conv2d<B: Backend>(path: &str, device: &B::Device) -> Result<conv::Conv2d<B>, Box<dyn Error>> {
    let weight = load_tensor::<B, 4>("weight", path, device)?;
    let bias = load_tensor::<B, 1>("bias", path, device).ok();
    let has_bias = bias.is_some();

    let stride = load_tensor::<B, 1>("stride", path, device)?;
    let stride = tensor_to_array_2(stride);

    let kernel_size = load_tensor::<B, 1>("kernel_size", path, device)?;
    let kernel_size = tensor_to_array_2(kernel_size);

    let dilation = load_tensor::<B, 1>("dilation", path, device)?;
    let dilation = tensor_to_array_2(dilation);

    let n_group = load_usize::<B>("n_group", path, device)?.into();
    let n_channels_in = load_usize::<B>("n_channels_in", path, device)?.into();
    let n_channels_out = load_usize::<B>("n_channels_out", path, device)?.into();

    let padding = load_tensor::<B, 1>("padding", path, device)?;
    let padding = tensor_to_array_2(padding);
    let padding = nn::PaddingConfig2d::Explicit(padding[0], padding[1]);

    
    let record = conv::Conv2dRecord {
        weight: weight.into(),
        bias: bias.map(|t| t.into()),
        stride: <[usize; 2] as Module<B>>::into_record(stride), 
        kernel_size: <[usize; 2] as Module<B>>::into_record(kernel_size), 
        dilation: <[usize; 2] as Module<B>>::into_record(dilation), 
        groups: <usize as Module<B>>::into_record(n_group),
        padding: <nn::PaddingConfig2d as Module<B>>::into_record(padding.clone()), 
    };

    let conv2d: conv::Conv2d<B> = conv::Conv2dConfig::new([n_channels_in, n_channels_out], kernel_size)
        .with_stride(stride)
        .with_dilation(dilation)
        .with_groups(n_group)
        .with_padding(padding)
        .with_bias(has_bias)
        .init_with(record);
    Ok(conv2d)
}

pub fn tensor_to_array_2<B: Backend>(x: Tensor<B, 1>) -> [usize; 2] {
    let vec = x.into_data().value;
    assert!(vec.len() == 2, "Tensor length must be 2.");
    [vec[0].to_usize().unwrap(), vec[1].to_usize().unwrap()]
}

pub fn tensor_to_array<const N: usize, B: Backend>(x: Tensor<B, 1>) -> [usize; N] {
    let vec = x.into_data().value;
    assert!(vec.len() == N, "Tensor length must be {}.", N);

    let mut arr = [0; N];
    for (a, t) in arr.iter_mut().zip(vec) {
        *a = t.to_usize().unwrap();
    }

    arr
}