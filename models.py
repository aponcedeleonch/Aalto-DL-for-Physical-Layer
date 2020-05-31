import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal
import math

# Models for tests on paper End-to-End Learning of Communications Systems Without a Channel Model

class Transmitter(nn.Module):
    """
    This is going to be the definition of the transmitter.
    """
    def __init__(self, m, n, embed_dim=512):
        """
        Initialization of the transmitter
        Quoting the paper:
        The transmitter consists of an MxM embedding with RELU activation functions,
        followed by a dense layer of 2N units with linear activations.
        This layer outputs 2N reals which are then converted into N complex symbols, and finally normalized
        
        Tensorflow dense = Pytorch Linear layer
        
        Args:
          m (int): Transmitter can have up to M different messages. Each of this messages gets encoded
          n (int): Length of the encoding
        """
        super(Transmitter, self).__init__()

        self.n = n
        
        self.transmit = nn.Sequential(
            nn.Embedding(num_embeddings=m, embedding_dim=m),
            nn.ReLU(),
            nn.Linear(in_features=m, out_features=2*n),
        )
        
#         # Different architecture tried. Leaving as comment.
#         self.transmit = nn.Sequential(
#             nn.Embedding(num_embeddings=m, embedding_dim=embed_dim),
#             nn.ReLU(),
#             nn.Linear(in_features=embed_dim, out_features=m),
#             nn.ReLU(),
#             nn.Linear(in_features=m, out_features=2*n),
#         )
        
        self.normalization = nn.BatchNorm1d(num_features=n)
        
        self.init_weights()
    
    def init_weights(self):
        """
        Function to initialize the weights and bias of the linear layers
        """
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)
        
    def forward(self, x):
        """
        Forward pass over x
        
        Args:
          x of shape (batch_size, 1): Index of the message to pass through the network
        
        Returns:
          x of shape (batch_size, n): Messages ready to be transmitted
        """
        batch_size = x.shape[0]
        
        x = self.transmit(x)
        # Conversion from 2N Real to N complex symbols
        x = x.reshape(batch_size, self.n, 2)
        x = self.normalization(x)
        
        return x

class Receiver(nn.Module):
    """
    This is going to be the definition of the receiver.
    """
    def __init__(self, m, n):
        """
        Initialization of the receiver
        Quoting the paper:
        Regarding the receiver, the first layer is a C2R layer which converts
        the received N complex symbols into 2N real symbols, while the last layer
        is a dense layer of M units with softmax activations which outputs a probability
        distribution over M
        
        Tensorflow dense = Pytorch Linear layer
        
        Args:
          m (int): Transmitter can have up to M different messages. Each of this messages gets encoded
          n (int): Length of the encoding
        """
        super(Receiver, self).__init__()

        self.n = n
        
        self.receive = nn.Sequential(
            nn.Linear(in_features=2*n, out_features=m),
            nn.ReLU(),
            nn.Linear(in_features=m, out_features=m),
            nn.LogSoftmax(dim=1),
        )
        
        self.estimate_h = nn.Sequential(
            nn.Linear(in_features=2*n, out_features=20),
            nn.Tanh(),
            nn.Linear(in_features=20, out_features=2),
        )
    
    def init_weights(self):
        """
        Function to initialize the weights and bias of the linear layers
        """
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)
        
    def forward(self, x, chann_type="AWGN"):
        """
        Forward pass over x
        
        Args:
          x of shape (batch_size, n, 2): Received messages
        
        Returns:
          x of shape (batch_size, m): Probabilities of element i in the batch corresponding to the mth message
        """
        batch_size = x.shape[0]
        
        if chann_type == "RBF":
            # Conversion from N complex symbols to 2N Real
            x_h = x.reshape(batch_size, 2*self.n)
            # Pass through the estimate h layer
            h_hat = self.estimate_h(x_h)
            h_hat = x_h.reshape(batch_size, 1, 2)
            
            # Divide original x by obtained h_hat
            x = x/h_hat

        # Conversion from N complex symbols to 2N Real
        x = x.reshape(batch_size, 2*self.n)
        x = self.receive(x)
        
        return x

    
class Policy(torch.nn.Module):
    """
    This is going to be the definition of the policy.
    """
    def __init__(self, m, n, sigma_var=0.02):
        """
        Initialization of the policy
        The action space of the policy are the n bits of the encoding.
        The policy has a static variance of sigma_var
        
        Args:
          m (int): Transmitter can have up to M different messages. Each of this messages gets encoded
          n (int): Length of the encoding
          sigma_var (float): Static variance of the policy
        """
        super(Policy, self).__init__()
        self.state_space = m
        self.action_space = n
        
        # Getting the variance and the standard deviation
        self.sigma_var = torch.tensor([sigma_var])
        self.sigma_std = torch.sqrt(self.sigma_var)

    def forward(self, x):
        """
        Forward pass over x in the policy
        
        Args:
          x of shape (batch_size, n, 2): Output of the transmitter
        
        Returns:
          xp of shape (batch_size, n, 2): xp of element i in the batch
          xp_logprob of shape (batch_size, n, 2): Log probability of xp over the policy
        """
        
        # Perturbation. Done in the paper to ensure exploration of the policy
        # Getting w to first scale x.
        w_dist = Normal(torch.tensor([0.0]), self.sigma_std)
        w = w_dist.sample(x.shape).to(x.device)
        # Get xp
        xp = torch.sqrt(1-self.sigma_var).to(x.device).detach()*x + w.squeeze()

        # Get the batch shize
        batch_size = x.shape[0]
        
        # Reshape all tensors currently in (batch_size, n, 2)
        # to easier shape to manage (batch_size, 2*n)
        x_dist = x.reshape(batch_size, 2*self.action_space)
        xp_dist = xp.reshape(batch_size, 2*self.action_space)
        
        # Log probabilities going to be stored here
        xp_logprob_dist = torch.zeros(batch_size, 2*self.action_space).to(x.device)
        
        # Getting the means for all the distributions
        policy_means = xp_dist.detach() - torch.sqrt(1-self.sigma_var).to(x.device).detach()*x_dist.detach()
        
        # Covariance matrix is the same for all the distributions
        cov_matrix = torch.eye(2*self.action_space).to(x.device) * self.sigma_var.to(x.device)
        
        for i in range(batch_size):
            # Get the log probability of each sample in xp
            xp_logprob_dist[i, :] = MultivariateNormal(policy_means[i], cov_matrix).log_prob(xp_dist[i])
        
        # Reshape back to (batch_size, n, 2) for standarization
        xp_logprob = xp_logprob_dist.reshape(batch_size, self.action_space, 2)

        # NB xp does NOT have a gradient. Log probabilities do
        return xp, xp_logprob  

# ---------------------------------------------------------------------- #

# Models for tests on paper An Introduction to Deep Learning for the Physical Layer


def paper_normalization(x, n):
    """
    The paper defines a specific normalization.
    Implementing it here to be able to use it Encoder
    """
    # Doing the normalization to get the final result of the encoder
    normalization_term = math.sqrt(n)/torch.sum(torch.sqrt(x**2), dim=1)
    x = x*normalization_term.unsqueeze(1)
    
    return x

class Encoder(nn.Module):
    """
    This is going to be the definition of the encoder.
    """
    def __init__(self, m, n, embed_dim=512, use_embedding=True, use_paper_norm=False):
        """
        The encoder in the paper is implemented with Tensorflow
        Layers in Tensorflow:
          - Dense + ReLU. Out M
              - Alternatively also use Embedding
          - Dense + linear. Out n
          - Normalization (some normalization defined by the paper) Out n
        Layers equivalent in Pytorch:
          - Linear + ReLU. Out M
              - Alternatively also use Embedding
          - Linear + (nothing). Out n
          - Normalization (some normalization defined by the paper or Pytorch normalization).
        
        Args:
          m (int): Transmitter can have up to M different messages. Each of this messages gets encoded
          n (int): Length of the encoding
        """
        super(Encoder, self).__init__()
        
        self.n = n
        self.use_paper_norm = use_paper_norm

        if use_embedding:
#             # Different architecture tried. Leaving as comment.
#             self.linear_M = nn.Sequential(
#                 nn.Embedding(num_embeddings=m, embedding_dim=embed_dim),
#                 nn.ReLU(),
#                 nn.Linear(in_features=embed_dim, out_features=m),
#                 nn.ReLU(),
#             )
            self.linear_M = nn.Sequential(
                nn.Embedding(num_embeddings=m, embedding_dim=m),
                nn.ReLU(),
            )
        else:
            self.linear_M = nn.Sequential(
                nn.Linear(in_features=m, out_features=m),
                nn.ReLU(),
            )
        
        self.linear_N = nn.Sequential(
            nn.Linear(in_features=m, out_features=n),
        )
        
        if use_paper_norm:
            self.normalization = paper_normalization
        else:
            self.normalization = nn.BatchNorm1d(num_features=n)
        
        self.init_weights()
    
    def init_weights(self):
        """
        Function to initialize the weights and bias of the linear layers
        """
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)
        
    def forward(self, x):
        """
        Forward pass over x
        
        Args:
          x of shape (batch_size, m): Messages that pass through the autoencoder
        
        Returns:
          y of shape (batch_size, n): Messages after encoding+noise+decoding
        """
        x = self.linear_M(x)
        x = self.linear_N(x.squeeze())
        if self.use_paper_norm:
            y = self.normalization(x, self.n)
        else:
            y = self.normalization(x)

        return y


class Decoder(nn.Module):
    """
    This is going to be the definition of the decoder.
    """
    def __init__(self, m, n):
        """
        Initialization of the decoder
        The decoder in the paper is implemented with Tensorflow
        Layers in Tensorflow:
          - Dense + ReLU. Out M
          - Dense + Softmax. Out M
        Layers equivalent in Pytorch:
          - Linear + ReLU. Out M
          - Linear + LogSoftmax. Out M
        LogSoftmax works well with Pytorch function nll_loss. That calculates the Negative Log Likelihood
        
        Args:
          m (int): Transmitter can have up to M different messages. Each of this messages gets encoded
          n (int): Length of the encoding
        """
        super(Decoder, self).__init__()
        
        self.linear_relu = nn.Sequential(
            nn.Linear(in_features=n, out_features=m),
            nn.ReLU(),
        )
        
        self.linear_out = nn.Sequential(
            nn.Linear(in_features=m, out_features=m),
            nn.LogSoftmax(dim=1),
        )
        
        self.init_weights()
        
    def init_weights(self):
        """
        Function to initialize the weights and bias of the linear layers
        """
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)
        
    def forward(self, y, chann_type="AWGN"):
        """
        Forward pass over y
        
        Args:
          y of shape (batch_size, n): Messages that have already passed through Encoder
        
        Returns:
          y of shape (batch_size, M): Messages after encoding+noise+decoding
        """
        # Decoding phase
        y = self.linear_relu(y)
        y = self.linear_out(y)
        
        return y