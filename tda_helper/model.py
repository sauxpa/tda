import abc
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .networks import Autoencoder, FeedForward
from .persistence import persistence_pairs
from .utils import triangular_from_linear_index, linear_index_from_triangular

    
class ModelHom(nn.Module):
    """Neural network model with homology penalization.
    """
    def __init__(self,
                 config_layers,
                 batch_size=50,
                 use_cuda=False,
                 lr=0.01,
                 tol=1e-6,
                 homology_dim=0,
                 homology_penalty=0.0,
                 homology_eps=0.0,
                 normalize_for_homology=False,
                 target_penalty=1.0,
                 compactness_penalty=0.0,
                 norm=1,
                 throttle=1,
                ):
        super(ModelHom, self).__init__()

        self.use_cuda = use_cuda
        self.batch_size = batch_size
        
        # config dict for underlying model
        self.config_layers = config_layers

        # numerical precision for distance lookup
        self.tol = tol
        # homology dimension
        self.homology_dim = homology_dim
        # weight for homology loss during training
        # positive : minimize persitence interval lengths
        # negative : maximize persitence interval lengths
        self.homology_penalty = homology_penalty
        # If True, compute persistent homology on the 
        # centered and scaled point cloud
        self.normalize_for_homology = normalize_for_homology
        # add one eps_hom to the homology loss for each 
        # persistent pair, in order to penalize not only the 
        # interval length but also the number of such intervals.
        self.homology_eps = homology_eps
        
        # weight for supervised target loss during training
        self.target_penalty = target_penalty
        
        # norm for point cloud radius penalization
        self.norm = norm
        # weight for compactness loss during training
        self.compactness_penalty = compactness_penalty
        
        # learning rate
        self.lr = lr

        # used for caching during training
        self.pdist = None
        self.persistence_births = None
        self.persistence_deaths = None
        
        # throttle tqdm updates
        self.throttle = throttle
        
    @property
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() and self.use_cuda else 'cpu')
    
    def indicator_birth(self, idx):
        """Returns True if idx corresponds to a pair of points the distance of which
        corresponds to critical filtration value in the Vietoris-Rips complex.
        """
        k = linear_index_from_triangular(self.batch_size, idx[0], idx[1])
        return True in torch.isclose(self.pdist[k], self.persistence_births, self.tol)
    
    def indicator_death(self, idx):
        """Returns True if idx corresponds to a pair of points the distance of which
        corresponds to critical filtration value in the Vietoris-Rips complex.
        """
        k = linear_index_from_triangular(self.batch_size, idx[0], idx[1])
        return True in torch.isclose(self.pdist[k], self.persistence_deaths, self.tol)

    @abc.abstractmethod
    def train(self, data, n_epochs):
        pass
    

class FeedForwardHom(ModelHom):
    """Feedforward network with homology penalization.
    """
    def __init__(self,
                 config_layers,
                 batch_size=50,
                 use_cuda=False,
                 lr=0.01,
                 tol=1e-6,
                 homology_dim=0,
                 homology_penalty=0.0,
                 homology_eps=0.0,
                 normalize_for_homology=False,
                 target_penalty=1.0,
                 activation='ReLU',
                 compactness_penalty=0.0,
                 norm=1,
                 throttle=1,
                ):
        super(FeedForwardHom, self).__init__(
            config_layers,
            batch_size=batch_size,
            use_cuda=use_cuda,
            lr=lr,
            tol=tol,
            homology_dim=homology_dim,
            homology_penalty=homology_penalty,
            homology_eps=homology_eps,
            normalize_for_homology=normalize_for_homology,
            target_penalty=target_penalty,
            compactness_penalty=compactness_penalty,
            norm=norm,
            throttle=throttle,
        )

        self.nn = FeedForward(
            input_size=self.config_layers['input_size'],
            hidden_size=self.config_layers['hidden_size'],
            n_layers=self.config_layers['n_layers'],
            output_size=self.config_layers['output_size'],
            activation=self.config_layers['activation'],
            bias=self.config_layers.get('bias', True),
        ).to(self.device)        
        
        parameters = list(self.nn.parameters())
        self.optimizer = torch.optim.Adam(parameters, lr=self.lr)

    def train(self, data, n_epochs):
        loader = torch.utils.data.DataLoader(data, batch_size=self.batch_size, shuffle=True)

        tdqm_dict_keys = ['homology', 'compactness', 'target']
        tdqm_dict = dict(zip(tdqm_dict_keys, [0.0, 0.0, 0,0]))

        for epoch in range(n_epochs):
            # initialize cumulative losses to zero at the start of epoch
            total_homology_loss = 0.0
            total_target_loss = 0.0
            total_compactness_loss = 0.0

            with tqdm(total=len(loader),
                      unit_scale=True,
                      postfix={'homology': 0.0, 
                               'compactness': 0.0, 
                               'target': 0.0},
                      desc="Epoch : %i/%i" % (epoch+1, n_epochs),
                      ncols=100
                     ) as pbar:
                for batch_idx, (batch, target) in enumerate(loader):
                    batch = batch.type(torch.float32).to(self.device)

                    y = self.nn(batch)
                    
                    if self.normalize_for_homology == 'std':
                        y_hom = (y-torch.mean(y, axis=0))/torch.std(y, axis=0)
                    elif self.normalize_for_homology == '01':
                        y_hom = (y-torch.min(y, axis=0).values)/(torch.max(y, axis=0).values-torch.min(y, axis=0).values)
                    else:
                        y_hom = y
                        
                    # in pure reconstruction mode, 
                    # skip the Gudhi part to speed training up
                    if self.homology_penalty == 0.0:
                        homology_loss = torch.FloatTensor([0.0]).to(self.device)
                    else:
                        # calculate pairwise distance matrix
                        # pdist is a flat tensor representing
                        # the upper triangle of the pairwise
                        # distance tensor.
                        self.pdist = F.pdist(y_hom)
                        # compute the persistence interval lengths
                        self.persistence_births, self.persistence_deaths = persistence_pairs(
                            y_hom,
                            dim=self.homology_dim, 
                            device=self.device
                        )
                        # Compute the indicator of indices that correspond
                        # to pairs of points such that the intersection of their
                        # balls in the Vietoris-Rips scheme is a birth or death event 
                        # for the homology of interest.
                        indicators_death = torch.FloatTensor(
                            [
                                self.indicator_death(triangular_from_linear_index(
                                    self.batch_size, k)
                                           ) for k in range(self.pdist.shape[0])]
                        ).to(self.device)

                        indicators_birth = torch.FloatTensor(
                            [
                                self.indicator_birth(triangular_from_linear_index(
                                    self.batch_size, k)
                                           ) for k in range(self.pdist.shape[0])]
                        ).to(self.device)

                        death_pdist = self.pdist[
                            torch.where(indicators_death == 1)[0]
                        ].to(self.device)
                        if self.homology_dim == 0:
                            birth_pdist = torch.zeros(death_pdist.shape).to(self.device)
                        else:
                            birth_pdist = self.pdist[
                                torch.where(indicators_birth == 1)[0]
                            ].to(self.device)
                        
                        # Due to rounding, it may sometimes happen that more pairs are located
                        # in the pdist matrix than there are : this hack bypasses that.
                        n_pairs = min([len(self.persistence_births), len(self.persistence_deaths)])
                        
                        # Compute homology loss
                        homology_loss = torch.norm(
                            death_pdist[:n_pairs]-birth_pdist[:n_pairs]+self.homology_eps, 
                            p=self.norm
                        )
                        
                    # MSE loss between true input and decoder output
                    target_loss = F.mse_loss(target, y)
                    
                    # A trivial solution to the homology optimization is to
                    # increase or decrease the scale of the latent point cloud.
                    # To avoid that, add a penalization on the radius of 
                    # the point cloud for a given norm.
                    compactness_loss = torch.norm(
                        y-torch.mean(y, axis=0), 
                        p=self.norm
                    )
                    
                    loss = self.target_penalty * target_loss \
                    + self.homology_penalty * homology_loss \
                    + self.compactness_penalty * compactness_loss

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    
                    total_homology_loss += homology_loss.item()
                    total_target_loss += target_loss.item()
                    total_compactness_loss += compactness_loss.item()
                    
                    # Logging
                    tdqm_dict['homology'] = total_homology_loss/(batch_idx+1)
                    tdqm_dict['target'] = total_target_loss/(batch_idx+1)
                    tdqm_dict['compactness'] = total_compactness_loss/(batch_idx+1)
                    if batch_idx % self.throttle == 0:
                        pbar.set_postfix(tdqm_dict)
                        pbar.update(self.throttle)
                    
                    
class AutoencoderHom(ModelHom):
    """Autoencoder with homology penalization.
    """
    def __init__(self,
                 config_layers,
                 batch_size=50,
                 use_cuda=False,
                 lr=0.01,
                 tol=1e-6,
                 homology_dim=0,
                 homology_penalty=0,
                 homology_eps=0.0,
                 normalize_for_homology=False,
                 target_penalty=1.0,
                 activation='ReLU',
                 compactness_penalty=0.0,
                 norm=1,
                 throttle=1,
                ):
        super(AutoencoderHom, self).__init__(
            config_layers,
            batch_size=batch_size,
            use_cuda=use_cuda,
            lr=lr,
            tol=tol,
            homology_dim=homology_dim,
            homology_penalty=homology_penalty,
            homology_eps=homology_eps,
            normalize_for_homology=normalize_for_homology,
            target_penalty=target_penalty,
            compactness_penalty=compactness_penalty,
            norm=norm,
            throttle=throttle,
        )
        
        # dim of latent space
        self.emb_size = self.config_layers['emb_size']
        
        self.autoencoder = Autoencoder(
            self.config_layers['input_size'],
            self.config_layers['hidden_size_enc'],
            self.config_layers['n_layers_enc'],
            self.config_layers['emb_size'],
            self.config_layers['hidden_size_dec'],
            self.config_layers['n_layers_dec'],
            self.config_layers['activation'],
            self.config_layers.get('bias', True),
        ).to(self.device)        

        parameters = list(self.autoencoder.parameters())
        self.optimizer = torch.optim.Adam(parameters, lr=self.lr)

    def train(self, data, n_epochs):
        loader = torch.utils.data.DataLoader(data, batch_size=self.batch_size, shuffle=True)

        tdqm_dict_keys = ['homology', 'compactness', 'reconstruction']
        tdqm_dict = dict(zip(tdqm_dict_keys, [0.0, 0.0, 0,0]))

        for epoch in range(n_epochs):
            # initialize cumulative losses to zero at the start of epoch
            total_homology_loss = 0.0
            total_reconstruction_loss = 0.0
            total_compactness_loss = 0.0

            with tqdm(total=len(loader),
                      unit_scale=True,
                      postfix={'homology': 0.0, 
                               'compactness': 0.0, 
                               'reconstruction': 0.0},
                      desc="Epoch : %i/%i" % (epoch+1, n_epochs),
                      ncols=100
                     ) as pbar:
                for batch_idx, batch in enumerate(loader):
                    batch = batch.type(torch.float32).to(self.device)

                    latent = self.autoencoder.encoder(batch)
                    
                    if self.normalize_for_homology == 'std':
                        latent_hom = (latent-torch.mean(latent, axis=0))/torch.std(latent, axis=0)
                    elif self.normalize_for_homology == '01':
                        latent_hom = (latent-torch.min(latent, axis=0).values)/(torch.max(latent, axis=0).values-torch.min(latent, axis=0).values)
                    else:
                        latent_hom = latent
                        
                    # in pure reconstruction mode, 
                    # skip the Gudhi part to speed training up
                    if self.homology_penalty == 0.0:
                        homology_loss = torch.FloatTensor([0.0]).to(self.device)
                    else:
                        # calculate pairwise distance matrix
                        # pdist is a flat tensor representing
                        # the upper triangle of the pairwise
                        # distance tensor.
                        self.pdist = F.pdist(latent_hom)
                        # compute the persistence interval lengths
                        self.persistence_births, self.persistence_deaths = persistence_pairs(
                            latent_hom,
                            dim=self.homology_dim, 
                            device=self.device
                        )
                        # Compute the indicator of indices that correspond
                        # to pairs of points such that the intersection of their
                        # balls in the Vietoris-Rips scheme is a birth or death event 
                        # for the homology of interest.
                        indicators_death = torch.FloatTensor(
                            [
                                self.indicator_death(triangular_from_linear_index(
                                    self.batch_size, k)
                                           ) for k in range(self.pdist.shape[0])]
                        ).to(self.device)

                        indicators_birth = torch.FloatTensor(
                            [
                                self.indicator_birth(triangular_from_linear_index(
                                    self.batch_size, k)
                                           ) for k in range(self.pdist.shape[0])]
                        ).to(self.device)

                        death_pdist = self.pdist[
                            torch.where(indicators_death == 1)[0]
                        ].to(self.device)
                        if self.homology_dim == 0:
                            birth_pdist = torch.zeros(death_pdist.shape).to(self.device)
                        else:
                            birth_pdist = self.pdist[
                                torch.where(indicators_birth == 1)[0]
                            ].to(self.device)
                        
                        # Due to rounding, it may sometimes happen that more pairs are located
                        # in the pdist matrix than there are : this hack bypasses that.
                        n_pairs = min([len(self.persistence_births), len(self.persistence_deaths)])
                        
                        # Compute homology loss
                        homology_loss = torch.norm(
                            death_pdist[:n_pairs]-birth_pdist[:n_pairs]+self.homology_eps, 
                            p=self.norm
                        )
                        
                    # MSE loss between true input and decoder output
                    reconstruction_loss = F.mse_loss(
                        batch, self.autoencoder.decoder(latent)
                    )
                    
                    # A trivial solution to the homology optimization is to
                    # increase or decrease the scale of the latent point cloud.
                    # To avoid that, add a penalization on the radius of 
                    # the point cloud for a given norm.
                    compactness_loss = torch.norm(
                        latent-torch.mean(latent, axis=0), 
                        p=self.norm
                    )

                    loss = self.target_penalty * reconstruction_loss \
                    + self.homology_penalty * homology_loss \
                    + self.compactness_penalty * compactness_loss

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    
                    total_homology_loss += homology_loss.item()
                    total_reconstruction_loss += reconstruction_loss.item()
                    total_compactness_loss += compactness_loss.item()
                    
                    # Logging
                    tdqm_dict['homology'] = total_homology_loss/(batch_idx+1)
                    tdqm_dict['reconstruction'] = total_reconstruction_loss/(batch_idx+1)
                    tdqm_dict['compactness'] = total_compactness_loss/(batch_idx+1)
                    
                    if batch_idx % self.throttle == 0:
                        pbar.set_postfix(tdqm_dict)
                        pbar.update(self.throttle)
