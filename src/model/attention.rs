use burn::{
    tensor::{
        backend::Backend,
        activation::softmax, 
        Tensor,
    },
};

use std::f32::NEG_INFINITY;

pub fn qkv_attention<B: Backend>(q: Tensor<B, 3>, k: Tensor<B, 3>, v: Tensor<B, 3>, mask: Option<Tensor<B, 2>>, n_head: usize) -> Tensor<B, 3> {
    let [n_batch, n_qctx, n_state] = q.dims();
    let [_, n_ctx, _] = k.dims();

    let scale = (n_state as f64 / n_head as f64).powf(-0.25);
    let n_hstate = n_state / n_head;

    let q = q.reshape([n_batch, n_qctx, n_head, n_hstate]).swap_dims(1, 2) * scale;
    let k = k.reshape([n_batch, n_ctx, n_head, n_hstate]).swap_dims(1, 2).transpose() * scale;
    let v = v.reshape([n_batch, n_ctx, n_head, n_hstate]).swap_dims(1, 2);

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

pub fn attn_decoder_mask<B: Backend>(seq_length: usize, device: &B::Device) -> Tensor<B, 2> {
    let mut mask = Tensor::<B, 2>::zeros([seq_length, seq_length]);

    for i in 0..(seq_length - 1) {
        let values = Tensor::<B, 2>::zeros([1, seq_length - (i + 1)]).add_scalar(NEG_INFINITY);
        mask = mask.slice_assign([i..i + 1, i + 1..seq_length], values);
    }

    return mask.to_device(device);
}