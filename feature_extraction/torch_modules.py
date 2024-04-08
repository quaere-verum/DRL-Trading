import torch as th
from gymnasium import spaces
from torch import nn
import numpy as np
import torch.nn.functional as F


class SeriesNet(nn.Module):
    '''
    Sub-module of CombinedLSTMExtractor. Processes a multivariate timeseries through an LSTM network. Subsequently\
    processes the LSTM hidden state and cell state through a CNN, to extract features and flatten these states. Finally,\
    vector outputs are concatenated and passed through a fully connected network.
    '''
    def __init__(self, device, **kwargs):
        super().__init__()
        self.device = device
        self.input_shape = kwargs['input_shape']
        self.lstm_layer = nn.LSTM(self.input_shape[1], kwargs['lstm_hidden_dim'], kwargs['lstm_num_layers'],
                                  batch_first=True).to(self.device)
        self.h_conv = nn.Conv2d(1, 1, kernel_size=(kwargs['lstm_num_layers'], 1), stride=1).to(self.device)
        self.c_conv = nn.Conv2d(1, 1, kernel_size=(kwargs['lstm_num_layers'], 1), stride=1).to(self.device)
        self.h_fc_layer = nn.Linear(kwargs['lstm_hidden_dim'], kwargs['h_fc_dim']).to(self.device)
        self.c_fc_layer = nn.Linear(kwargs['lstm_hidden_dim'], kwargs['c_fc_dim']).to(self.device)
        self.fc_layers = [
            nn.Linear(kwargs['lstm_hidden_dim'] + kwargs['c_fc_dim'] + kwargs['h_fc_dim'], kwargs['fc_arch'][0]).to(
                self.device)]
        self.fc_layers.extend([nn.Linear(kwargs['fc_arch'][k - 1], kwargs['fc_arch'][k]).to(self.device) for k in
                               range(1, len(kwargs['fc_arch']))])


    def forward(self, series):
        series = series.to(self.device)
        x, (h, c) = self.lstm_layer(series)
        y = F.relu(self.h_conv(h.transpose(0, 1).unsqueeze(1)).squeeze(1, 2))
        y = F.relu(self.h_fc_layer(y))
        z = F.relu(self.c_conv(c.transpose(0, 1).unsqueeze(1)).squeeze(1, 2))
        z = F.relu(self.c_fc_layer(z))
        x = th.cat((x[:, -1, :], y, z), dim=-1)
        for layer in self.fc_layers:
            x = F.relu(layer(x))
        return x


class CombinedLSTMExtractor(nn.Module):
    '''
    Features extractor designed for a custom gymnasium environment whose observation space is an instance of spaces.Dict,\
    and whose keys are "prices", "sequential" and "other".

    First the prices and sequential features are passed through separate instances of SeriesNet. Outputs are concatenated\
    with remaining features and passed through a fully connected network.
    '''

    def __init__(self, obs_space, prices_net_kwargs, sequential_net_kwargs, fc_arch, device=None):
        super().__init__()
        self.prices_dim = obs_space['prices'].shape
        self.sequential_dim = obs_space['sequential'].shape
        self.other_dim = obs_space['other'].shape[0]

        self.features_dim = fc_arch[-1] #Necessary for usage as a feature extractor in stable_baselines3
        self.device = device

        self.prices_net = SeriesNet(self.device, **prices_net_kwargs).to(self.device)
        self.sequential_net = SeriesNet(self.device, **sequential_net_kwargs).to(self.device)
        self.fc_layers = [
            nn.Linear(prices_net_kwargs['fc_arch'][-1] + sequential_net_kwargs['fc_arch'][-1] + self.other_dim,
                      fc_arch[0]).to(self.device)]
        self.fc_layers.extend([nn.Linear(fc_arch[k - 1], fc_arch[k]).to(self.device) for k in range(1, len(fc_arch))])


    def forward(self, inputs):
        prices = inputs['prices'].view((-1,) + self.prices_dim).to(self.device)
        sequential = inputs['sequential'].view((-1,) + self.sequential_dim).to(self.device)
        other = inputs['other'].view(-1, self.other_dim).to(self.device)
        x = th.cat((self.prices_net(prices), self.sequential_net(sequential), other), dim=-1)
        for layer in self.fc_layers:
            x = F.relu(layer(x))
        return x

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, input_shape, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Parameter(th.empty(1, input_shape[0], input_shape[1]))
        nn.init.uniform_(self.pe, -0.02, 0.02)

    def forward(self, x):
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


class TSTransformerEncoderClassiregressor(nn.Module):
    def __init__(self, input_shape, d_model, n_heads, num_layers, dim_feedforward, output_dim=1,
                 dropout=0.1, pos_encoding='fixed', activation='gelu', norm='BatchNorm', freeze=False):
        super().__init__()

        self.input_shape = input_shape
        self.d_model = d_model
        self.n_heads = n_heads

        self.input_projection = nn.Linear(input_shape[1], d_model)
        self.positional_encoding = LearnablePositionalEncoding((input_shape[0], d_model),
                                                               dropout=dropout * (1.0 - freeze))

        encoder_layer = nn.TransformerEncoderLayer(d_model, self.n_heads, dim_feedforward, dropout * (1.0 - freeze),
                                                   activation=activation, batch_first=True)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers, enable_nested_tensor=False)

        self.dropout1 = nn.Dropout(dropout)

        self.feat_dim = input_shape[1]
        self.output_layer = nn.Linear(d_model * input_shape[0], output_dim)

    def forward(self, x, padding_masks):
        x = self.input_projection(x) * np.sqrt(self.d_model)
        x = self.positional_encoding(x)
        x = F.gelu(self.transformer_encoder(x, src_key_padding_mask=~padding_masks))
        x = self.dropout1(x)
        x = x * padding_masks.unsqueeze(-1)
        x = x.reshape(x.shape[0], -1)
        x = self.output_layer(x)

        return x