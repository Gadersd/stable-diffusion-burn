use burn::tensor::{activation::softmax, Tensor};

pub trait Backend: burn::tensor::backend::Backend {
    fn qkv_attention(
        q: Self::TensorPrimitive<3>,
        k: Self::TensorPrimitive<3>,
        v: Self::TensorPrimitive<3>,
        mask: Option<Self::TensorPrimitive<2>>,
        n_head: usize,
    ) -> Self::TensorPrimitive<3> {
        qkv_attention(
            Tensor::<Self, 3>::from_primitive(q),
            Tensor::from_primitive(k),
            Tensor::from_primitive(v),
            mask.map(|m| Tensor::from_primitive(m)),
            n_head,
        )
        .into_primitive()
    }

    fn attn_decoder_mask(seq_length: usize, device: &Self::Device) -> Self::TensorPrimitive<2> {
        attn_decoder_mask::<Self>(seq_length, device).into_primitive()
    }
}

use burn::tensor::ops::TensorOps;
use burn::tensor::Float;
use burn_tch::{self, TchElement, TchTensor};
use tch;

impl<E: TchElement> Backend for burn_tch::LibTorch<E> {
    fn qkv_attention(
        q: Self::TensorPrimitive<3>,
        k: Self::TensorPrimitive<3>,
        v: Self::TensorPrimitive<3>,
        mask: Option<Self::TensorPrimitive<2>>,
        n_head: usize,
    ) -> Self::TensorPrimitive<3> {
        let q = Tensor::from_primitive(q);
        let k = Tensor::from_primitive(k);
        let v = Tensor::from_primitive(v);

        let [n_batch, q_ctx, n_state] = q.dims();
        let [_, k_ctx, _] = k.dims();
        let n_hstate = n_state / n_head;

        let rearrange = |t: Tensor<Self, 3>| {
            let [_, n_ctx, _] = t.dims();
            t.reshape([n_batch, n_ctx, n_head, n_hstate])
                .swap_dims(1, 2)
        };

        let q = rearrange(q).into_primitive();
        let k = rearrange(k).into_primitive();
        let v = rearrange(v).into_primitive();

        // for some reason torch crashes when mask is None
        let mask = mask.unwrap_or_else(|| {
            Tensor::<Self, 2, Float>::zeros_device([q_ctx, k_ctx], &Self::device(&v))
                .into_primitive()
        });

        Tensor::<Self, 4>::from_primitive(TchTensor::new(
            tch::Tensor::scaled_dot_product_attention(
                &q.tensor,
                &k.tensor,
                &v.tensor,
                Some(mask.tensor),
                0.0,
                false,
                None
            ),
        ))
        .swap_dims(1, 2)
        .flatten(2, 3)
        .into_primitive()
    }
}

use burn_autodiff;

impl<B: Backend> Backend for burn_autodiff::Autodiff<B> {}

use std::f32::NEG_INFINITY;

fn qkv_attention<B: Backend>(
    q: Tensor<B, 3>,
    k: Tensor<B, 3>,
    v: Tensor<B, 3>,
    mask: Option<Tensor<B, 2>>,
    n_head: usize,
) -> Tensor<B, 3> {
    let [n_batch, n_qctx, n_state] = q.dims();
    let [_, n_ctx, _] = k.dims();

    let scale = (n_state as f64 / n_head as f64).powf(-0.25);
    let n_hstate = n_state / n_head;

    let q = q
        .reshape([n_batch, n_qctx, n_head, n_hstate])
        .swap_dims(1, 2)
        * scale;
    let k = k
        .reshape([n_batch, n_ctx, n_head, n_hstate])
        .swap_dims(1, 2)
        .transpose()
        * scale;
    let v = v
        .reshape([n_batch, n_ctx, n_head, n_hstate])
        .swap_dims(1, 2);

    let qk = q.matmul(k);

    // apply mask
    let qk = if let Some(mask) = mask {
        qk + mask.slice([0..n_qctx, 0..n_ctx]).unsqueeze::<4>()
    } else {
        qk
    };

    // normalize value weightings
    let w = softmax(qk, 3);
    let o = w.matmul(v).swap_dims(1, 2).flatten(2, 3);

    return o;
}

fn attn_decoder_mask<B: Backend>(seq_length: usize, device: &B::Device) -> Tensor<B, 2> {
    let mut mask = Tensor::<B, 2>::zeros([seq_length, seq_length]);

    for i in 0..(seq_length - 1) {
        let values = Tensor::<B, 2>::zeros([1, seq_length - (i + 1)]).add_scalar(NEG_INFINITY);
        mask = mask.slice_assign([i..i + 1, i + 1..seq_length], values);
    }

    return mask.to_device(device);
}
