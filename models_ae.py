# Import necessary modules and functions
from functools import wraps
import numpy as np
import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from torch_cluster import fps
from timm.models.layers import DropPath

# Check if a value exists
def exists(val):
    return val is not None

# Return default value if the given value doesn't exist
def default(val, d):
    return val if exists(val) else d

# Define a decorator function to cache the result of another function
def cache_fn(f):
    # Initialize cache as None
    cache = None
    
    # Use wraps to preserve the metadata of the original function
    @wraps(f)
    def cached_fn(*args, _cache = True, **kwargs):
        # If caching is disabled, directly call the original function
        if not _cache:
            return f(*args, **kwargs)
        
        # Use nonlocal to modify the outer cache variable
        nonlocal cache
        
        # If cache exists, return the cached result
        if cache is not None:
            return cache
        
        # If cache doesn't exist, call the original function and store the result
        cache = f(*args, **kwargs)
        
        # Return the newly cached result
        return cache
    
    # Return the wrapper function
    return cached_fn

# Pre-normalization module
class PreNorm(nn.Module):
    # Initialize the PreNorm module
    def __init__(self, dim, fn, context_dim = None):
        super().__init__()
        self.fn = fn  # Store the function to be applied after normalization
        self.norm = nn.LayerNorm(dim)  # Create a LayerNorm for the input
        # Create a LayerNorm for the context if context_dim is provided
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    # Forward pass of the PreNorm module
    def forward(self, x, **kwargs):
        x = self.norm(x)  # Apply normalization to the input

        # If context normalization is available
        if exists(self.norm_context):
            context = kwargs['context']  # Get the context from kwargs
            normed_context = self.norm_context(context)  # Normalize the context
            kwargs.update(context = normed_context)  # Update kwargs with normalized context

        return self.fn(x, **kwargs)  # Apply the function to normalized input and return

# GELU activation with gating
class GEGLU(nn.Module):
    def forward(self, x):
        # Split the input tensor into two chunks along the last dimension
        x, gates = x.chunk(2, dim=-1)
        
        # Apply GELU activation to the gates and multiply with x
        # This creates a gating mechanism using the GELU function
        return x * F.gelu(gates)

# Feed-forward network module
class FeedForward(nn.Module):
    # Initialize the FeedForward module
    def __init__(self, dim, mult=4, drop_path_rate=0.0):
        super().__init__()
        # Define the feed-forward network
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),  # Expand the input dimension
            GEGLU(),  # Apply GELU activation with gating
            nn.Linear(dim * mult, dim)  # Project back to original dimension
        )

        # Set up drop path (stochastic depth) if rate > 0, otherwise use identity
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    # Define the forward pass
    def forward(self, x):
        # Apply the network and drop path
        return self.drop_path(self.net(x))

# Multi-head attention module
class Attention(nn.Module):
    def __init__(self, query_dim, context_dim = None, heads = 8, dim_head = 64, drop_path_rate = 0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, query_dim)

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x, context = None, mask = None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h = h)
            sim.masked_fill_(~mask, max_neg_value)

        # Compute attention weights and apply them to values
        attn = sim.softmax(dim = -1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        return self.drop_path(self.to_out(out))

# Point embedding module
class PointEmbed(nn.Module):
    def __init__(self, hidden_dim=48, dim=128):
        super().__init__()

        assert hidden_dim % 6 == 0

        self.embedding_dim = hidden_dim
        e = torch.pow(2, torch.arange(self.embedding_dim // 6)).float() * np.pi
        e = torch.stack([
            torch.cat([e, torch.zeros(self.embedding_dim // 6),
                        torch.zeros(self.embedding_dim // 6)]),
            torch.cat([torch.zeros(self.embedding_dim // 6), e,
                        torch.zeros(self.embedding_dim // 6)]),
            torch.cat([torch.zeros(self.embedding_dim // 6),
                        torch.zeros(self.embedding_dim // 6), e]),
        ])
        self.register_buffer('basis', e)  # 3 x 16

        self.mlp = nn.Linear(self.embedding_dim+3, dim)

    @staticmethod
    def embed(input, basis):
        projections = torch.einsum(
            'bnd,de->bne', input, basis)
        embeddings = torch.cat([projections.sin(), projections.cos()], dim=2)
        return embeddings
    
    def forward(self, input):
        # input: B x N x 3
        embed = self.mlp(torch.cat([self.embed(input, self.basis), input], dim=2)) # B x N x C
        return embed

class DiagonalGaussianDistribution(object):
    def __init__(self, mean, logvar, deterministic=False):
        self.mean = mean
        self.logvar = logvar
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.mean.device)

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.mean.device)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                return 0.5 * torch.mean(torch.pow(self.mean, 2)
                                       + self.var - 1.0 - self.logvar,
                                       dim=[1, 2])
            else:
                return 0.5 * torch.mean(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var - 1.0 - self.logvar + other.logvar,
                    dim=[1, 2, 3])

    def nll(self, sample, dims=[1,2,3]):
        if self.deterministic:
            return torch.Tensor([0.])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims)

    def mode(self):
        return self.mean

class AutoEncoder(nn.Module):
    def __init__(
        self,
        *,
        depth=24,
        dim=512,
        queries_dim=512,
        output_dim=1,
        num_inputs=2048,
        num_latents=512,
        heads=8,
        dim_head=64,
        weight_tie_layers=False,
        decoder_ff=False
    ):
        """
        Initialize the AutoEncoder.

        Args:
            depth (int): Number of layers in the encoder.
            dim (int): Dimension of the latent space.
            queries_dim (int): Dimension of the query embeddings.
            output_dim (int): Dimension of the output.
            num_inputs (int): Number of input points.
            num_latents (int): Number of latent points.
            heads (int): Number of attention heads.
            dim_head (int): Dimension of each attention head.
            weight_tie_layers (bool): Whether to tie weights across layers.
            decoder_ff (bool): Whether to use a feedforward layer in the decoder.
        """
        super().__init__()

        self.depth = depth
        self.num_inputs = num_inputs
        self.num_latents = num_latents

        # Cross attention blocks for initial encoding
        self.cross_attend_blocks = nn.ModuleList([
            PreNorm(dim, Attention(dim, dim, heads=1, dim_head=dim), context_dim=dim),
            PreNorm(dim, FeedForward(dim))
        ])

        # Point embedding layer
        self.point_embed = PointEmbed(dim=dim)

        # Define attention and feedforward layers for the encoder
        get_latent_attn = lambda: PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, drop_path_rate=0.1))
        get_latent_ff = lambda: PreNorm(dim, FeedForward(dim, drop_path_rate=0.1))
        get_latent_attn, get_latent_ff = map(cache_fn, (get_latent_attn, get_latent_ff))

        # Create encoder layers
        self.layers = nn.ModuleList([])
        cache_args = {'_cache': weight_tie_layers}
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                get_latent_attn(**cache_args),
                get_latent_ff(**cache_args)
            ]))

        # Decoder components
        self.decoder_cross_attn = PreNorm(queries_dim, Attention(queries_dim, dim, heads=1, dim_head=dim), context_dim=dim)
        self.decoder_ff = PreNorm(queries_dim, FeedForward(queries_dim)) if decoder_ff else None

        # Output layer
        self.to_outputs = nn.Linear(queries_dim, output_dim) if exists(output_dim) else nn.Identity()

    def encode(self, pc):
        """
        Encode the input point cloud.

        Args:
            pc (torch.Tensor): Input point cloud of shape (B, N, 3).

        Returns:
            torch.Tensor: Encoded latent representation.
        """
        B, N, D = pc.shape
        assert N == self.num_inputs

        # Farthest point sampling
        flattened = pc.view(B*N, D)
        batch = torch.arange(B).to(pc.device)
        batch = torch.repeat_interleave(batch, N)
        pos = flattened
        ratio = 1.0 * self.num_latents / self.num_inputs
        idx = fps(pos, batch, ratio=ratio)
        sampled_pc = pos[idx].view(B, -1, 3)

        # Embed sampled points and original points
        sampled_pc_embeddings = self.point_embed(sampled_pc)
        pc_embeddings = self.point_embed(pc)

        # Apply cross attention
        cross_attn, cross_ff = self.cross_attend_blocks
        x = cross_attn(sampled_pc_embeddings, context=pc_embeddings, mask=None) + sampled_pc_embeddings
        x = cross_ff(x) + x

        return x

    def decode(self, x, queries):
        """
        Decode the latent representation.

        Args:
            x (torch.Tensor): Latent representation.
            queries (torch.Tensor): Query points.

        Returns:
            torch.Tensor: Decoded output.
        """
        # Apply self-attention layers
        for self_attn, self_ff in self.layers:
            x = self_attn(x) + x
            x = self_ff(x) + x

        # Cross attend from decoder queries to latents
        queries_embeddings = self.point_embed(queries)
        latents = self.decoder_cross_attn(queries_embeddings, context=x)

        # Apply optional decoder feedforward
        if exists(self.decoder_ff):
            latents = latents + self.decoder_ff(latents)
        
        return self.to_outputs(latents)

    def forward(self, pc, queries):
        """
        Forward pass of the AutoEncoder.

        Args:
            pc (torch.Tensor): Input point cloud.
            queries (torch.Tensor): Query points.

        Returns:
            dict: Dictionary containing the output logits.
        """
        x = self.encode(pc)
        o = self.decode(x, queries).squeeze(-1)
        return {'logits': o}

# KL Divergence Autoencoder
class KLAutoEncoder(nn.Module):
    def __init__(
        self,
        *,
        depth=24,                # Number of transformer layers
        dim=512,                 # Dimension of the model
        queries_dim=512,         # Dimension of queries
        output_dim = 1,          # Output dimension
        num_inputs = 2048,       # Number of input points
        num_latents = 512,       # Number of latent points
        latent_dim = 64,         # Dimension of latent space
        heads = 8,               # Number of attention heads
        dim_head = 64,           # Dimension of each attention head
        weight_tie_layers = False,  # Whether to tie weights across layers
        decoder_ff = False       # Whether to use feedforward in decoder
    ):
        super().__init__()

        self.depth = depth
        self.num_inputs = num_inputs
        self.num_latents = num_latents

        # Cross attention blocks for initial processing
        self.cross_attend_blocks = nn.ModuleList([
            PreNorm(dim, Attention(dim, dim, heads = 1, dim_head = dim), context_dim = dim),
            PreNorm(dim, FeedForward(dim))
        ])

        # Embedding layer for point clouds
        self.point_embed = PointEmbed(dim=dim)

        # Lambda functions for creating attention and feedforward layers
        get_latent_attn = lambda: PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, drop_path_rate=0.1))
        get_latent_ff = lambda: PreNorm(dim, FeedForward(dim, drop_path_rate=0.1))
        get_latent_attn, get_latent_ff = map(cache_fn, (get_latent_attn, get_latent_ff))

        # Create transformer layers
        self.layers = nn.ModuleList([])
        cache_args = {'_cache': weight_tie_layers}

        for i in range(depth):
            self.layers.append(nn.ModuleList([
                get_latent_attn(**cache_args),
                get_latent_ff(**cache_args)
            ]))

        # Decoder components
        self.decoder_cross_attn = PreNorm(queries_dim, Attention(queries_dim, dim, heads = 1, dim_head = dim), context_dim = dim)
        self.decoder_ff = PreNorm(queries_dim, FeedForward(queries_dim)) if decoder_ff else None

        # Output layer
        self.to_outputs = nn.Linear(queries_dim, output_dim) if exists(output_dim) else nn.Identity()

        # Projection layer from latent space to model dimension
        self.proj = nn.Linear(latent_dim, dim)

        # Layers for mean and log variance in latent space
        self.mean_fc = nn.Linear(dim, latent_dim)
        self.logvar_fc = nn.Linear(dim, latent_dim)

    def encode(self, pc):
        # Encode input point cloud to latent representation
        # pc: B x N x 3 (B: batch size, N: number of points, 3: xyz coordinates)
        B, N, D = pc.shape
        assert N == self.num_inputs

        # Farthest Point Sampling
        flattened = pc.view(B*N, D)
        batch = torch.arange(B).to(pc.device)
        batch = torch.repeat_interleave(batch, N)
        pos = flattened
        ratio = 1.0 * self.num_latents / self.num_inputs
        idx = fps(pos, batch, ratio=ratio)
        sampled_pc = pos[idx]
        sampled_pc = sampled_pc.view(B, -1, 3)

        # Embed sampled points and original points
        sampled_pc_embeddings = self.point_embed(sampled_pc)
        pc_embeddings = self.point_embed(pc)

        # Apply cross attention
        cross_attn, cross_ff = self.cross_attend_blocks
        x = cross_attn(sampled_pc_embeddings, context = pc_embeddings, mask = None) + sampled_pc_embeddings
        x = cross_ff(x) + x

        # Generate mean and log variance for latent distribution
        mean = self.mean_fc(x)
        logvar = self.logvar_fc(x)

        # Sample from the latent distribution
        posterior = DiagonalGaussianDistribution(mean, logvar)
        x = posterior.sample()
        kl = posterior.kl()

        return kl, x

    def decode(self, x, queries):
        # Decode latent representation to output
        x = self.proj(x)

        # Apply transformer layers
        for self_attn, self_ff in self.layers:
            x = self_attn(x) + x
            x = self_ff(x) + x

        # Cross attend from decoder queries to latents
        queries_embeddings = self.point_embed(queries)
        latents = self.decoder_cross_attn(queries_embeddings, context = x)

        # Apply optional decoder feedforward
        if exists(self.decoder_ff):
            latents = latents + self.decoder_ff(latents)
        
        return self.to_outputs(latents)

    def forward(self, pc, queries):
        # Forward pass through the entire model
        kl, x = self.encode(pc)
        o = self.decode(x, queries).squeeze(-1)
        return {'logits': o, 'kl': kl}

def create_autoencoder(dim=512, M=512, latent_dim=64, N=2048, determinisitc=False):
    # Factory function to create either deterministic or KL autoencoder
    if determinisitc:
        model = AutoEncoder(
            depth=24,
            dim=dim,
            queries_dim=dim,
            output_dim = 1,
            num_inputs = N,
            num_latents = M,
            heads = 8,
            dim_head = 64,
        )
    else:
        model = KLAutoEncoder(
            depth=24,
            dim=dim,
            queries_dim=dim,
            output_dim = 1,
            num_inputs = N,
            num_latents = M,
            latent_dim = latent_dim,
            heads = 8,
            dim_head = 64,
        )
    return model

# Autoencoder model configurations

# KL Divergence Autoencoders with varying latent dimensions
def kl_d512_m512_l512(N=2048):
    """
    Create a KL Divergence Autoencoder with 512 latent dimensions.
    Args:
        N (int): Number of input points (default: 2048)
    Returns:
        KLAutoEncoder: Autoencoder model
    """
    return create_autoencoder(dim=512, M=512, latent_dim=512, N=N, determinisitc=False)

def kl_d512_m512_l64(N=2048):
    """
    Create a KL Divergence Autoencoder with 64 latent dimensions.
    Args:
        N (int): Number of input points (default: 2048)
    Returns:
        KLAutoEncoder: Autoencoder model
    """
    return create_autoencoder(dim=512, M=512, latent_dim=64, N=N, determinisitc=False)

def kl_d512_m512_l32(N=2048):
    """
    Create a KL Divergence Autoencoder with 32 latent dimensions.
    Args:
        N (int): Number of input points (default: 2048)
    Returns:
        KLAutoEncoder: Autoencoder model
    """
    return create_autoencoder(dim=512, M=512, latent_dim=32, N=N, determinisitc=False)

def kl_d512_m512_l16(N=2048):
    """
    Create a KL Divergence Autoencoder with 16 latent dimensions.
    Args:
        N (int): Number of input points (default: 2048)
    Returns:
        KLAutoEncoder: Autoencoder model
    """
    return create_autoencoder(dim=512, M=512, latent_dim=16, N=N, determinisitc=False)

def kl_d512_m512_l8(N=2048):
    """
    Create a KL Divergence Autoencoder with 8 latent dimensions.
    Args:
        N (int): Number of input points (default: 2048)
    Returns:
        KLAutoEncoder: Autoencoder model
    """
    return create_autoencoder(dim=512, M=512, latent_dim=8, N=N, determinisitc=False)

def kl_d512_m512_l4(N=2048):
    """
    Create a KL Divergence Autoencoder with 4 latent dimensions.
    Args:
        N (int): Number of input points (default: 2048)
    Returns:
        KLAutoEncoder: Autoencoder model
    """
    return create_autoencoder(dim=512, M=512, latent_dim=4, N=N, determinisitc=False)

def kl_d512_m512_l2(N=2048):
    """
    Create a KL Divergence Autoencoder with 2 latent dimensions.
    Args:
        N (int): Number of input points (default: 2048)
    Returns:
        KLAutoEncoder: Autoencoder model
    """
    return create_autoencoder(dim=512, M=512, latent_dim=2, N=N, determinisitc=False)

def kl_d512_m512_l1(N=2048):
    """
    Create a KL Divergence Autoencoder with 1 latent dimension.
    Args:
        N (int): Number of input points (default: 2048)
    Returns:
        KLAutoEncoder: Autoencoder model
    """
    return create_autoencoder(dim=512, M=512, latent_dim=1, N=N, determinisitc=False)

# Deterministic Autoencoders with varying M (number of latents)
def ae_d512_m512(N=2048):
    """
    Create a deterministic Autoencoder with 512 latents.
    Args:
        N (int): Number of input points (default: 2048)
    Returns:
        AutoEncoder: Autoencoder model
    """
    return create_autoencoder(dim=512, M=512, N=N, determinisitc=True)

def ae_d512_m256(N=2048):
    """
    Create a deterministic Autoencoder with 256 latents.
    Args:
        N (int): Number of input points (default: 2048)
    Returns:
        AutoEncoder: Autoencoder model
    """
    return create_autoencoder(dim=512, M=256, N=N, determinisitc=True)

def ae_d512_m128(N=2048):
    """
    Create a deterministic Autoencoder with 128 latents.
    Args:
        N (int): Number of input points (default: 2048)
    Returns:
        AutoEncoder: Autoencoder model
    """
    return create_autoencoder(dim=512, M=128, N=N, determinisitc=True)

def ae_d512_m64(N=2048):
    """
    Create a deterministic Autoencoder with 64 latents.
    Args:
        N (int): Number of input points (default: 2048)
    Returns:
        AutoEncoder: Autoencoder model
    """
    return create_autoencoder(dim=512, M=64, N=N, determinisitc=True)

# Deterministic Autoencoders with varying dimensions
def ae_d256_m512(N=2048):
    """
    Create a deterministic Autoencoder with dimension 256.
    Args:
        N (int): Number of input points (default: 2048)
    Returns:
        AutoEncoder: Autoencoder model
    """
    return create_autoencoder(dim=256, M=512, N=N, determinisitc=True)

def ae_d128_m512(N=2048):
    """
    Create a deterministic Autoencoder with dimension 128.
    Args:
        N (int): Number of input points (default: 2048)
    Returns:
        AutoEncoder: Autoencoder model
    """
    return create_autoencoder(dim=128, M=512, N=N, determinisitc=True)

def ae_d64_m512(N=2048):
    """
    Create a deterministic Autoencoder with dimension 64.
    Args:
        N (int): Number of input points (default: 2048)
    Returns:
        AutoEncoder: Autoencoder model
    """
    return create_autoencoder(dim=64, M=512, N=N, determinisitc=True)