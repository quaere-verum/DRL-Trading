import torch as th
import torch.nn.functional as F
from torch import nn
import numpy as np
import pandas as pd

def prepare_autoencoder_data(s, lookback_window, resample=True, normalise=True):
    '''
    :param s: pd.DataFrame containing the timeseries to be encoded
    :param lookback_window: int specifying the resampling or rolling window in minutes
    :param resample: bool specifying whether to resample, or to take a rolling window
    :param normalise: bool specifying whether to normalise each window
    :return: np.ndarray containing the normalised training data
    '''
    if resample:
        start = -(len(s)//lookback_window*lookback_window)
        X = np.stack([x[1].values for x in s.iloc[start:, :].resample(f'{lookback_window}min')], axis=0)
    else:
        X = np.flip((pd.DataFrame(s).shift(np.arange(lookback_window)).dropna().values.reshape(
            -1, lookback_window, s.shape[1])), axis=1)
    if normalise:
        divisor = np.std(X, axis=1).reshape(-1, 1, s.shape[1])
        return (X - np.mean(X, axis=1).reshape(-1, 1, s.shape[1])) / np.where(divisor > 0, divisor, 1)
    return X


class TSE(nn.Module):
    '''
    Network architecture: input is a batch of B multivariate timeseries of length T, so input shape is (B, T, M) where
    M is the number of distinct timeseries. This is passed through a Conv2d layer whose kernel size is enforced to be
    (k, M), so the network "summarises" the information of the k timesteps that the kernel is applied to. Next, max
    pooling is applied and the out_channels are univariate timeseries which are subsequently passed into Conv1d layers
    for further data compression. The output of the convolutional layers is then flattened and passed through a fully
    connected layer, whose output is the encoded timeseries.
    '''
    def __init__(self, input_shape, encoder_conv2d_arch,
                 encoder_conv1d_arch=[{'out_channels': 32, 'kernel_size': 4, 'stride': 2},
                                      {'out_channels': 16, 'kernel_size': 5, 'stride': 2}],
                 max_pool_arch={'stride': 1, 'kernel_size': 10}, fc_arch=(128, 64), output_dim=32,
                 decoder_arch=(128, 128), device=None):
        '''
        Implements a TimeSeries Encoder (TSE). One aspect of timeseries data is that it is sequential in nature, and that
        each step (generally) changes the value only slightly. If we want to use standard machine learning models like
        XGBoost, it makes little sense (conceptually) to feed direct timeseries data as input. Instead, we would like to
        encode the timeseries into some vector space such the vector representing a timeseries can be used to reconstruct
        it. For example, some dimensions of this vector space might represent short term characteristics of the timeseries,
        whether other dimensions might represent long term characteristics. This kind of input is more suitable for
        XGBoost models, feedforward neural networks, and can be used in combination with transformer blocks and/or
        multi-headed attention.
        :param input_shape: (series length, num_features)
        :param encoder_conv2d_arch: dictionary containing out_channels, kernel_size and stride
        :param encoder_conv1d_arch: dictionary containing out_channels, kernel_size and stride
        :param max_pool_arch: dictionary containing kernel_size and stride
        :param fc_arch: tuple containing the nunber of neurons in the final MLP layers
        :param output_dim: dimensionality of the encoding space
        :param decoder_arch: tuple containing the number of neurons in each MLP layer of the decoder
        :param device: th.device
        '''
        super().__init__()
        assert encoder_conv2d_arch['kernel_size'][1] == input_shape[1]
        self.input_shape = input_shape
        self.output_dim = 32
        self.device = device
        self.layer_norm = nn.LayerNorm(normalized_shape=input_shape).to(device)
        self.max_pool = nn.MaxPool1d(**max_pool_arch).to(device)
        self.encoder_conv2d_layer = nn.Conv2d(in_channels=1, **encoder_conv2d_arch).to(device)
        self.encoder_conv1d_layers = [
            nn.Conv1d(in_channels=encoder_conv2d_arch['out_channels'], padding=0, **encoder_conv1d_arch[0]).to(device)]
        self.encoder_conv1d_layers.extend([nn.Conv1d(in_channels=encoder_conv1d_arch[k - 1]['out_channels'], padding=0,
                                                     **encoder_conv1d_arch[k]).to(device) for k in
                                           range(1, len(encoder_conv1d_arch))])

        strides = int(
            np.ceil((input_shape[0] - encoder_conv2d_arch['kernel_size'][0] + 1) / encoder_conv2d_arch['stride']))
        strides = int(np.ceil((strides - max_pool_arch['kernel_size'] + 1) / max_pool_arch['stride']))
        for layer in encoder_conv1d_arch:
            strides = int(np.ceil((strides - layer['kernel_size'] + 1) / layer['stride']))
        assert strides > 0
        self.encoder_fc_layers = [nn.Linear(strides * encoder_conv1d_arch[-1]['out_channels'], fc_arch[0]).to(device)]
        self.encoder_fc_layers.extend(
            [nn.Linear(fc_arch[k - 1], fc_arch[k]).to(device) for k in range(1, len(fc_arch))])
        self.encoder_output_layer = nn.Linear(fc_arch[-1], output_dim).to(device)

        self.decoder_entry_layer = nn.Linear(output_dim, decoder_arch[0]).to(device)
        self.decoder_heads = []
        for head in range(input_shape[1]):
            modules = []
            for k in range(1, len(decoder_arch)):
                modules.extend([nn.Linear(decoder_arch[k - 1], decoder_arch[k]), nn.ReLU()])
            modules.extend([nn.Linear(decoder_arch[-1], input_shape[0]), nn.Tanh()])
            self.decoder_heads.append(nn.Sequential(*modules).to(device))

    def forward_encoder(self, x):
        x = x.to(self.device)
        x = self.layer_norm(x).unsqueeze(1)
        x = F.relu(self.encoder_conv2d_layer(x).squeeze(-1))
        x = self.max_pool(x)
        for layer in self.encoder_conv1d_layers:
            x = F.relu(layer(x))
        x = x.view(x.shape[0], -1)
        for layer in self.encoder_fc_layers:
            x = F.relu(layer(x))
        return self.encoder_output_layer(x)

    def forward_decoder(self, x):
        x = x.to(self.device)
        x = F.relu(self.decoder_entry_layer(x))
        return th.stack([layer(x) for layer in self.decoder_heads], dim=-1)

    def forward(self, x):
        x = self.forward_encoder(x)
        return self.forward_decoder(x)