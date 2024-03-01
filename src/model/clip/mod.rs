pub mod load;

use burn::{
    config::Config,
    module::{Module, Param},
    nn,
    tensor::{
        activation::{sigmoid, softmax},
        backend::Backend,
        module::embedding,
        Distribution, Int, Tensor,
    },
};

use crate::backend::Backend as MyBackend;

#[derive(Config)]
pub struct CLIPConfig {
    n_vocab: usize,
    n_state: usize,
    n_head: usize,
    n_ctx: usize,
    n_layer: usize,
}

impl CLIPConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> CLIP<B> {
        let token_embedding = nn::EmbeddingConfig::new(self.n_vocab, self.n_state).init(device);
        let position_embedding = Tensor::random(
            [self.n_ctx, self.n_state],
            Distribution::Normal(0.0, 1.0),
            device,
        )
        .into();
        let blocks = (0..self.n_layer)
            .into_iter()
            .map(|_| {
                ResidualDecoderAttentionBlockConfig::new(self.n_state, self.n_head).init(device)
            })
            .collect();
        let layer_norm = nn::LayerNormConfig::new(self.n_state).init(device);

        CLIP {
            token_embedding,
            position_embedding,
            blocks,
            layer_norm,
        }
    }
}

#[derive(Module, Debug)]
pub struct CLIP<B: Backend> {
    token_embedding: nn::Embedding<B>,
    position_embedding: Param<Tensor<B, 2>>,
    blocks: Vec<ResidualDecoderAttentionBlock<B>>,
    layer_norm: nn::LayerNorm<B>,
}

impl<B: MyBackend> CLIP<B> {
    pub fn forward(&self, x: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [n_batch, seq_len] = x.dims();

        let mask = Tensor::from_primitive(B::attn_decoder_mask(seq_len, &x.device()));

        let embedded = self.token_embedding.forward(x)
            + self
                .position_embedding
                .val()
                .slice([0..seq_len])
                .unsqueeze();

        let mut x = embedded;
        for block in &self.blocks {
            x = block.forward(x, mask.clone());
        }

        self.layer_norm.forward(x)
    }
}

#[derive(Config)]
pub struct ResidualDecoderAttentionBlockConfig {
    n_state: usize,
    n_head: usize,
}

impl ResidualDecoderAttentionBlockConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> ResidualDecoderAttentionBlock<B> {
        let attn = MultiHeadSelfAttentionConfig::new(self.n_state, self.n_head).init(device);
        let attn_ln = nn::LayerNormConfig::new(self.n_state).init(device);

        let mlp = MLPConfig::new(self.n_state, 4 * self.n_state).init(device);
        let mlp_ln = nn::LayerNormConfig::new(self.n_state).init(device);

        ResidualDecoderAttentionBlock {
            attn,
            attn_ln,
            mlp,
            mlp_ln,
        }
    }
}

#[derive(Module, Debug)]
pub struct ResidualDecoderAttentionBlock<B: Backend> {
    attn: MultiHeadSelfAttention<B>,
    attn_ln: nn::LayerNorm<B>,
    mlp: MLP<B>,
    mlp_ln: nn::LayerNorm<B>,
}

impl<B: MyBackend> ResidualDecoderAttentionBlock<B> {
    fn forward(&self, x: Tensor<B, 3>, mask: Tensor<B, 2>) -> Tensor<B, 3> {
        let x = x.clone() + self.attn.forward(self.attn_ln.forward(x), Some(mask));
        let x = x.clone() + self.mlp.forward(self.mlp_ln.forward(x));
        return x;
    }
}

#[derive(Config)]
pub struct MultiHeadSelfAttentionConfig {
    n_state: usize,
    n_head: usize,
}

impl MultiHeadSelfAttentionConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> MultiHeadSelfAttention<B> {
        assert!(
            self.n_state % self.n_head == 0,
            "State size {} must be a multiple of head size {}",
            self.n_state,
            self.n_head
        );

        let n_head = self.n_head;
        let query = nn::LinearConfig::new(self.n_state, self.n_state).init(device);
        let key = nn::LinearConfig::new(self.n_state, self.n_state).init(device);
        let value = nn::LinearConfig::new(self.n_state, self.n_state).init(device);
        let out = nn::LinearConfig::new(self.n_state, self.n_state).init(device);

        MultiHeadSelfAttention {
            n_head,
            query,
            key,
            value,
            out,
        }
    }
}

#[derive(Module, Debug)]
pub struct MultiHeadSelfAttention<B: Backend> {
    n_head: usize,
    query: nn::Linear<B>,
    key: nn::Linear<B>,
    value: nn::Linear<B>,
    out: nn::Linear<B>,
}

impl<B: MyBackend> MultiHeadSelfAttention<B> {
    pub fn forward(&self, x: Tensor<B, 3>, mask: Option<Tensor<B, 2>>) -> Tensor<B, 3> {
        let q = self.query.forward(x.clone());
        let k = self.key.forward(x.clone());
        let v = self.value.forward(x.clone());

        let device = x.device().clone();
        let wv = Tensor::from_primitive(B::qkv_attention(
            q.into_primitive(),
            k.into_primitive(),
            v.into_primitive(),
            mask.map(|m| m.into_primitive()),
            self.n_head,
            &device,
        ));

        return self.out.forward(wv);
    }
}

#[derive(Config, Debug)]
pub struct MLPConfig {
    input_size: usize,
    hidden_size: usize,
}

impl MLPConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> MLP<B> {
        let fc1 = nn::LinearConfig::new(self.input_size, self.hidden_size).init(device);
        let gelu = QuickGELU::new();
        let fc2 = nn::LinearConfig::new(self.hidden_size, self.input_size).init(device);

        MLP { fc1, gelu, fc2 }
    }
}

#[derive(Module, Debug)]
pub struct MLP<B: Backend> {
    fc1: nn::Linear<B>,
    gelu: QuickGELU,
    fc2: nn::Linear<B>,
}

impl<B: Backend> MLP<B> {
    fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        let x = self.fc1.forward(x);
        let x = self.gelu.forward(x);
        let x = self.fc2.forward(x);

        x
    }
}

#[derive(Module, Clone, Debug)]
pub struct QuickGELU {}

impl QuickGELU {
    fn new() -> Self {
        Self {}
    }

    fn forward<B: Backend, const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        x.clone() * sigmoid(x * 1.702)
    }
}
