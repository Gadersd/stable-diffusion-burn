pub mod load;

use burn::{
    config::Config,
    module::{Module, Param},
    tensor::{backend::Backend, Tensor},
};

#[derive(Config)]
pub struct GroupNormConfig {
    n_group: usize,
    n_channel: usize,
    #[config(default = 1e-5)]
    eps: f64,
}

impl GroupNormConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> GroupNorm<B> {
        assert!(
            self.n_channel % self.n_group == 0,
            "The number of channels {} must be divisible by the number of groups {}",
            self.n_channel,
            self.n_group
        );

        let n_per_group = self.n_channel / self.n_group;

        let gamma = Tensor::ones([self.n_channel], device).into();
        let beta = Tensor::zeros([self.n_channel], device).into();

        let eps = self.eps;

        GroupNorm {
            n_group: self.n_group,
            n_channel: self.n_channel,
            gamma,
            beta,
            eps,
        }
    }
}

#[derive(Module, Debug)]
pub struct GroupNorm<B: Backend> {
    n_group: usize,
    n_channel: usize,
    gamma: Param<Tensor<B, 1>>,
    beta: Param<Tensor<B, 1>>,
    eps: f64,
}

impl<B: Backend> GroupNorm<B> {
    pub fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        let shape = x.shape();
        let n_batch = shape.dims[0];
        let num_elements = shape.num_elements();

        let mut affine_shape = [1; D];
        affine_shape[1] = self.n_channel;

        layernorm(
            x.reshape([
                n_batch,
                self.n_group,
                num_elements / (n_batch * self.n_group),
            ]),
            self.eps,
        )
        .reshape(shape)
        .mul(self.gamma.val().reshape(affine_shape))
        .add(self.beta.val().reshape(affine_shape))
    }
}

pub fn layernorm<B: Backend, const D: usize>(x: Tensor<B, D>, eps: f64) -> Tensor<B, D> {
    //let (var, mean) = x.clone().var_mean_bias(D - 1);
    //x.sub(mean).div(var.sqrt().add_scalar(eps))

    let u = x.clone() - x.mean_dim(D - 1);
    u.clone()
        .div((u.clone() * u).mean_dim(D - 1).add_scalar(eps).sqrt())
}
