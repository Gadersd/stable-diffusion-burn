import pathlib
import save
from save import *

def save_clipmlp(clip_mlp, path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    save_linear(clip_mlp.fc1, pathlib.Path(path, 'fc1'))
    save_linear(clip_mlp.fc2, pathlib.Path(path, 'fc2'))

def save_clip_attention(clip_attention, path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    save_linear(clip_attention.k_proj, pathlib.Path(path, 'key'))
    save_linear(clip_attention.v_proj, pathlib.Path(path, 'value'))
    save_linear(clip_attention.q_proj, pathlib.Path(path, 'query'))
    save_linear(clip_attention.out_proj, pathlib.Path(path, 'out'))
    save_scalar(clip_attention.num_heads, 'n_head', path)

def save_clip_encoder_layer(clip_encoder_layer, path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    save_clip_attention(clip_encoder_layer.self_attn, pathlib.Path(path, 'attn'))
    save_layer_norm(clip_encoder_layer.layer_norm1, pathlib.Path(path, 'attn_ln'))
    save_clipmlp(clip_encoder_layer.mlp, pathlib.Path(path, 'mlp'))
    save_layer_norm(clip_encoder_layer.layer_norm2, pathlib.Path(path, 'mlp_ln'))

def save_clip_encoder(clip_encoder, path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    for i, layer in enumerate(clip_encoder.layers):
        save_clip_encoder_layer(layer, pathlib.Path(path, f'blocks/{i}'))
    save_scalar(len(clip_encoder.layers), "n_layer", path)

def save_clip_text_embeddings(clip_text_embeddings, path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    save_embedding(clip_text_embeddings.token_embedding, pathlib.Path(path, 'token_embedding'))
    save_embedding(clip_text_embeddings.position_embedding, pathlib.Path(path, 'position_embedding'))

def save_clip_text_transformer(clip_text_transformer, path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    save_clip_text_embeddings(clip_text_transformer.embeddings, path)
    save_clip_encoder(clip_text_transformer.encoder, path)
    save_layer_norm(clip_text_transformer.final_layer_norm, pathlib.Path(path, 'layer_norm'))