import torch
from torch.distributions.negative_binomial import NegativeBinomial as NB
from torch.distributions.normal import Normal  
from torch.nn import Linear, Module, functional as F



class Encoder(Module):
    
    def __init__(
            self, 
            n_genes: int, 
            n_variants: int, 
            n_gene_signatures: int, 
            n_variant_signatures: int,
            latent_dim: int,
        ) -> None:
        super().__init__()

        self.reduce_variants = Linear(n_variants, n_variant_signatures)
        self.reduce_genes = Linear(n_genes, n_gene_signatures)
        self.fc_mean  = Linear(n_gene_signatures * n_variant_signatures, latent_dim)
        self.fc_logvar = Linear(n_gene_signatures * n_variant_signatures, latent_dim)
                
        self.training = True
        
    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        reduced_variants = F.relu(self.reduce_variants(x_in)).transpose(-2, -1)
        reduced_genes = F.relu(self.reduce_genes(reduced_variants)).flatten(-2, -1)
        
        mean = self.fc_mean(reduced_genes)
        logvar = self.fc_logvar(reduced_genes)                                                 
        
        return mean, logvar
    

class Decoder(Module):
    def __init__(
            self, 
            n_genes: int, 
            n_variants: int, 
            n_gene_signatures: int, 
            n_variant_signatures: int,
            latent_dim: int,
        ) -> None:        
        super().__init__()

        self.n_gene_signatures = n_gene_signatures
        self.n_variant_signatures = n_variant_signatures
        self.latent_dim = latent_dim

        self.hidden_layer = Linear(latent_dim, n_gene_signatures * n_variant_signatures)
        self.inflate_genes = Linear(n_gene_signatures, n_genes)
        self.inflate_variants = Linear(n_variant_signatures, n_variants)
        self.reconstruction_p = Linear(latent_dim, 1)
                
    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        hidden = F.relu(self.hidden_layer(latent))
        hidden = hidden.view(hidden.shape[:-1] + (self.n_variant_signatures, self.n_gene_signatures))
        inflated_genes = F.relu(self.inflate_genes(hidden)).transpose(-2, -1)
        
        reconstruction_r = torch.exp(self.inflate_variants(inflated_genes))
        reconstruction_p = self.reconstruction_p(latent).squeeze(-1)
        
        return reconstruction_r, reconstruction_p
    
    def sample_from_latent(self, latent: torch.Tensor) -> torch.Tensor:
        reconstruction_r, reconstruction_p = self.forward(latent)
        reconstruction_p = reconstruction_p.unsqueeze(-1).repeat(reconstruction_r.shape)
        return NB(reconstruction_r, logits=reconstruction_p).sample()

    def generate(self) -> torch.Tensor:
        latent = Normal(loc=[0.] * self.latent_dim, scale = [1.] * self.latent_dim).sample()
        return self.sample_from_latent(latent)
    
class NBVAE(Module):
    def __init__(
        self,
        n_genes: int,
        n_variants: int, 
        n_gene_signatures: int = 1, 
        n_variant_signatures: int = 1, 
        latent_dim: int = 3
    ) -> None:
        super().__init__()
        self.encoder = Encoder(
            n_genes=n_genes, 
            n_variants=n_variants, 
            n_gene_signatures=n_gene_signatures, 
            n_variant_signatures=n_variant_signatures, 
            latent_dim=latent_dim
            )
        self.decoder = Decoder(
            n_genes=n_genes, 
            n_variants=n_variants, 
            n_gene_signatures=n_gene_signatures, 
            n_variant_signatures=n_variant_signatures, 
            latent_dim=latent_dim
        )

    def reparameterise(self, mean: torch.Tensor, var: torch.Tensor):
        epsilon = torch.randn_like(var)
        return mean + var * epsilon
    
    def get_latent(self, x_in: torch.Tensor):
        mean, logvar = self.encoder(x_in)
        latent = self.reparameterise(mean, torch.exp(0.5 * logvar))
        return latent, mean, logvar

    def forward(self, x_in: torch.Tensor):
        latent, mean, logvar = self.get_latent(x_in)
        reconstruction_r, reconstruction_p = self.decoder(latent)
        
        return reconstruction_r, reconstruction_p, mean, logvar
    
    def sample_from_input(self, x_in: torch.Tensor):
        mean, _ = self.encoder(x_in)
        return self.decoder.sample_from_latent(mean).to_sparse_coo()

def kullback_liebler_divergence(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return - 0.5 * (1 + logvar - mean.pow(2) - logvar.exp()).sum(dim=-1).mean()

def nb_log_likelihood(x_in: torch.Tensor, reconstruction_r: torch.Tensor, reconstruction_p: torch.Tensor) -> torch.Tensor:
    ll = torch.where(x_in > 0, torch.lgamma(x_in + reconstruction_r) - torch.lgamma(reconstruction_r), 0).sum(dim=(-2, -1))
    ll += reconstruction_p * x_in.sum(dim=(-2,-1)) 
    ll -= torch.log(torch.exp(reconstruction_p) + 1) * (x_in + reconstruction_r).sum(dim=(-2, -1))
    return ll.mean()

def nbvae_loss(x_in: torch.Tensor, reconstruction_r: torch.Tensor, reconstruction_p: torch.Tensor, mean: torch.Tensor, logvar: torch.Tensor, kld_weight: float = 1.):
    nb_ll = - nb_log_likelihood(x_in, reconstruction_r, reconstruction_p).mean()
    kld = kullback_liebler_divergence(mean, logvar).mean()
    return nb_ll + kld_weight * kld 
