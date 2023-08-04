import pathlib
import numpy as np

from tinygrad.tensor import Tensor

def save_scalar(s, name, path):
    s = np.array([1.0, float(s)]).astype(np.float32)
    np.save(pathlib.Path(path, f'{name}.npy'), s)

def save_tensor(tensor, name, path):
    tensor_numpy = tensor.numpy()
    tensor_dims = np.array(tensor_numpy.shape)
    tensor_values = tensor_numpy.flatten()
    tensor_to_save = np.concatenate((tensor_dims, tensor_values)).astype(np.float32)
    np.save(pathlib.Path(path, f'{name}.npy'), tensor_to_save)

def save_linear(linear, path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    save_tensor(linear.weight.transpose(), 'weight', path) # PyTorch and Tinygrad strangely transpose linear weights so reverse that
    if linear.bias is not None:
        save_tensor(linear.bias, 'bias', path)

def save_layer_norm(layer_norm, path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    save_tensor(layer_norm.weight, 'weight', path)
    save_tensor(layer_norm.bias, 'bias', path)
    save_scalar(layer_norm.eps, 'eps', path)

def save_group_norm(layer_norm, path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    if layer_norm.weight is not None:
        save_tensor(layer_norm.weight, 'weight', path)
    if layer_norm.bias is not None:
        save_tensor(layer_norm.bias, 'bias', path)
    save_scalar(layer_norm.eps, 'eps', path)
    save_scalar(layer_norm.num_groups, 'n_group', path)
    save_scalar(layer_norm.num_channels, 'n_channel', path)

def to_tuple_tensor(val):
    if isinstance(val, tuple):
        # Convert tuple to Tensor
        if len(val) == 1:
            return Tensor([val[0], val[0]])
        elif len(val) == 2: 
            return Tensor([val[0], val[1]])
        else:
            raise ValueError('Tuple should be of length 1 or 2 only.')
    else:
        # Treat as scalar and convert to Tensor
        return Tensor([val, val])

def save_conv2d(conv2d, path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    
    save_tensor(conv2d.weight, 'weight', path)
    if conv2d.bias is not None:
        save_tensor(conv2d.bias, 'bias', path)
    save_tensor(to_tuple_tensor(conv2d.stride), 'stride', path)
    save_tensor(to_tuple_tensor(conv2d.padding), 'padding', path)
    save_tensor(to_tuple_tensor(conv2d.dilation), 'dilation', path)
    save_scalar(conv2d.groups, "n_group", path)
    save_tensor(to_tuple_tensor(conv2d.kernel_size), 'kernel_size', path)

    assert conv2d.groups == 1
    in_channels = conv2d.weight.shape[1]
    out_channels = conv2d.weight.shape[0]
    save_scalar(in_channels, "n_channels_in", path)
    save_scalar(out_channels, "n_channels_out", path)

def save_padded_conv2d(padded_conv2d, path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)

    # Store conv2d layer weights
    orig_padding = padded_conv2d.padding
    padded_conv2d.padding = (0, 0)
    save_conv2d(padded_conv2d, f"{path}/conv")
    padded_conv2d.padding = orig_padding

    # Dimensions: in-channels and out-channels
    assert padded_conv2d.groups == 1
    channels = (padded_conv2d.weight.shape[1], padded_conv2d.weight.shape[0])
    save_tensor(to_tuple_tensor(channels), 'channels', path)

    assert len(padded_conv2d.kernel_size) == 1 or padded_conv2d.kernel_size[0] == padded_conv2d.kernel_size[1]
    save_scalar(padded_conv2d.kernel_size[0], 'kernel_size', path) 

    # Stride
    assert not isinstance(padded_conv2d.stride, tuple) or len(padded_conv2d.stride) == 1
    save_scalar(padded_conv2d.stride, 'stride', path)

    # Padding
    padding = [padded_conv2d.padding[0], padded_conv2d.padding[1],
               padded_conv2d.padding[2], padded_conv2d.padding[3]]
    save_tensor(Tensor(padding), 'padding', path)


def save_embedding(embedding, path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    save_tensor(embedding.weight, 'weight', path)

