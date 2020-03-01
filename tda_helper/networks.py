import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    """Basic customizable network structure
    """
    def __init__(self,
                 hidden_size=2,
                 n_layers=1,
                 input_size=1,
                 output_size=1,
                 activation='ReLU',
                 bias=True,
                ):
        super(FeedForward, self).__init__()
        
        self.n_layers = n_layers
        
        if self.n_layers == 1:
            self.layers = [nn.Linear(input_size, output_size, bias=bias)]
        else:
            size  = [input_size] + [hidden_size,] * (self.n_layers-1) + [output_size]
            self.layers = [nn.Linear(size[i], size[i+1], bias=bias) for i in range(self.n_layers)]
        
        # stack all layers
        self.layers = nn.ModuleList(self.layers)
        
        # activation
        if activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'ReLU':
            self.activation = nn.ReLU()
        elif activation == 'LeakyReLU':
            self.activation = nn.LeakyReLU(negative_slope=0.1)
        else:
            raise Exception('{} not an available activation'.format(activation))
        
    def forward(self, x):
        for i in range(self.n_layers-1):
            x = self.activation(self.layers[i](x))
        x = self.layers[-1](x)
        return x
    

class Encoder(FeedForward):
    def __init__(self, 
                 input_size, 
                 hidden_size, 
                 emb_size, 
                 n_layers=1, 
                 activation='ReLU', 
                 bias=True,
                ):
        super(Encoder, self).__init__(
            hidden_size=hidden_size,
            n_layers=n_layers,
            input_size=input_size,
            output_size=emb_size,
            activation=activation,
            bias=bias,
        )

        
class Decoder(FeedForward):
    def __init__(self, 
                 emb_size, 
                 hidden_size, 
                 output_size, 
                 n_layers=1, 
                 activation='ReLU', 
                 bias=True,
                ):
        super(Decoder, self).__init__(
            hidden_size=hidden_size,
            n_layers=n_layers,
            input_size=emb_size,
            output_size=output_size,
            activation=activation,
            bias=bias,
        )
        
        
class Autoencoder(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size_enc,
                 n_layers_enc,
                 emb_size,
                 hidden_size_dec,
                 n_layers_dec,
                 activation='ReLU',
                 bias=True,
                ):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(input_size, 
                               hidden_size_enc, 
                               emb_size, 
                               n_layers_enc, 
                               activation, 
                               bias
                              )
        self.decoder = Decoder(emb_size, 
                               hidden_size_dec, 
                               input_size, 
                               n_layers_dec, 
                               activation, 
                               bias
                              )

    def forward(self, x):
        return self.decoder(self.encoder(x))
