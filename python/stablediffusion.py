from autoencoder import save_autoencoder
from unet import save_unet_model
from clip import save_clip_text_transformer

from save import save_scalar, save_tensor

def save_stable_diffusion(stable_diffusion, path):
    save_scalar(stable_diffusion.alphas_cumprod.shape[0], "n_steps", path)
    save_tensor(stable_diffusion.alphas_cumprod, 'alphas_cumprod', path)
    save_autoencoder(stable_diffusion.autoencoder, 'autoencoder', path)
    save_unet_model(stable_diffusion.diffusion, 'unet', path)
    save_clip_text_transformer(stable_diffusion.clip, 'clip', path)