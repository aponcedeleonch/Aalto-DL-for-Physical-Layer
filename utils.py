import torch
import numpy as np
import math

# To make plots about constellations
from sklearn.manifold import TSNE
import matplotlib.cm as cm

from models import Transmitter, Receiver, Encoder, Decoder
from comms_utils import channel

MODELS_FOLDER = 'trained_models'

def plot_constellation(m, n, ax, device, model="supervised", chann_type="AWGN", snr_db=10, samples=20):
    """
    Function to plot a constellation with an already trained autoencoder
    Args:
        n (int): Length of the encoded messages
        m ((int): Total number of messages that can be encoded
        ax (matplotlib object): Plotting in a subplot. Received subplot object
        snr_db (float): Noise to add to the encoded messages
        samples (int): Number of samples to plot
        model (string): Model that wish to be recovered. Options: supervised or alternated
        chann_type (string): Channel type. Currently only AWGN available
    Returns:
        ax (matplotlib object): Plot completed
    """
    k = math.log2(m)
    enc, _ = recover_models(device, model=model, m=m, n=n, chann_type=chann_type)

    with torch.no_grad():
        data = torch.arange(0, m).to(device)
        enc_data = enc(data)

    # Pass from pytorch to numpy
    if model == "alternated":
        enc_numpy = enc_data.reshape(m, 2*n).to("cpu").detach().numpy()
    else:
        enc_numpy = enc_data.to("cpu").detach().numpy()

    # We have m different messages
    classes = np.arange(m)

    # To store the results of each sample
    if model == "alternated":
        results_noise = np.zeros((1, 2*n))
    else:
        results_noise = np.zeros((1, n))
    
    labels = np.zeros(1)
    
    for i in range(samples):
        # Pass the encoded messages through noise
        noise_encode = channel(enc_numpy, n, k, snr_db, chann_type=chann_type)
        # Add the samples to the array
        results_noise = np.concatenate((results_noise, noise_encode), axis=0)
        labels = np.concatenate((labels, classes))
    
    # The first row was a dummy to use concatenate. Removing it
    results_noise = results_noise[1:]
    labels = labels[1:]

    # Getting a 2 dimensional embedding for the points
    x_constellation = TSNE(n_components=2).fit_transform(results_noise)

    # Generating as many colors as neededd
    colors = cm.rainbow(np.linspace(0, 1, len(classes)))
    
    # Plotting each different message
    for i in classes:
        ix = labels == i
        ax.plot(x_constellation[ix, 0], x_constellation[ix, 1], '.', color=colors[i])

    ax.set_title("Constellation for noise %s. %s training" % (chann_type, model.capitalize()))

    return ax


def count_errors(inputs, targets):
    """
    Function to try count the errrors after Rx/Decoding wrt original messages (targets)
    Args:
        inputs pytorch tensor of shape(batch_size, m): 
        targets pytorch tensor of shape(batch_size): 
    Returns:
        total_errrors (float): Total errors found
    """
    # Each example i has m probabilities. Is the probabilit of example i being m message
    # Choose the highest probability
    chosen_input = torch.argmax(inputs, dim=1)
    
    # Get where both tensors are different
    errors = targets != chosen_input
    
    # Sum the errors to get the total
    total_errors = errors.sum().to("cpu").numpy()
    
    return total_errors


class MemoryMessages():
    """
    Small class to get samples at every epoch during training
    """
    def __init__(self, m, use_embedding=True):
        """
        Intialize the class
        Args:
            m (int): Up to m possible messages
            use_embedding (boolean): If we are using embedding or one-hot encoded vectors
        """
        # Initialize memory
        self.memory = np.arange(m)
        self.use_embedding = use_embedding
        self.m = m
        
    def __len__(self):
        """
        To return the length remaining memory
        Returns:
            (int) memory size
        """
        return len(self.memory)

    def sample(self, batch_size=32):
        """
        Sample a batch of batch_size from memory
        Args:
            batch_size (int): Size of batch to sample
        Returns:
            batch: sampled batch
            targets: targets of the sampled batch
        """
        batch = []
        targets = []
        
        # Get batch_size samples
        for i in range(batch_size):
            # If we still have memory keep sampling
            if len(self.memory) > 0:
                # Get a random index from the memory
                idx = np.random.randint(0, len(self.memory))
                
                targets.append(self.memory[idx])
                
                # If we use embedding we return an index (int)
                # If we are not using embedding we return a one-hot encoded vector
                if self.use_embedding:
                    batch.append(self.memory[idx])
                else:
                    vec_onehot = np.zeros((self.m, ))
                    vec_onehot[idx] = 1
                    batch.append(vec_onehot)
                
                # Delete the sampled element from memory
                self.memory = np.delete(self.memory, idx)
            else:
                return np.array(batch), np.array(targets)
        
        return np.array(batch), np.array(targets)

    
def recover_models(device, model="supervised", m=256, n=4, chann_type="AWGN", verbose=False):
    """
    Function to try to recover an already saved system to a channel
    Args:
        device (string): Current device that we are working in
        model (string): Model that wish to be recovered. Options: supervised or alternated
        chann_type (string): Channel type. Currently only AWGN available
        n (int): Length of the encoded messages
        m ((int): Total number of messages that can be encoded
    Returns:
        encoder/tx (Object): Recovered Tx/Encoder model
        decoder/rx (Object): Recovered Rx/Decoder model
    """
    try:
        if model == "supervised":
            enc_filename = "%s/%s_%d_%d_encoder.pth" % (MODELS_FOLDER, chann_type, m, n)
            dec_filename = "%s/%s_%d_%d_decoder.pth" % (MODELS_FOLDER, chann_type, m, n)

            encoder = Encoder(m=m, n=n)
            encoder.load_state_dict(torch.load(enc_filename))
            if verbose: print('Model loaded from %s.' % enc_filename)
            # Put them in the correct device and eval mode
            encoder.to(device)
            encoder.eval()

            decoder = Decoder(m=m, n=n)
            decoder.load_state_dict(torch.load(dec_filename))
            if verbose: print('Model loaded from %s.' % dec_filename)
            decoder.to(device)
            decoder.eval()

            return encoder, decoder
        else:
            tx_filename = "%s/%s_%d_%d_tx.pth" % (MODELS_FOLDER, chann_type, m, n)
            rx_filename = "%s/%s_%d_%d_rx.pth" % (MODELS_FOLDER, chann_type, m, n)

            tx = Transmitter(m=m, n=n)
            tx.load_state_dict(torch.load(tx_filename))
            if verbose: print('Model loaded from %s.' % tx_filename)
            # Put them in the correct device and eval mode
            tx.to(device)
            tx.eval()

            rx = Receiver(m=m, n=n)
            rx.load_state_dict(torch.load(rx_filename))
            if verbose: print('Model loaded from %s.' % rx_filename)
            rx.to(device)
            rx.eval()
        
            return tx, rx
    except:
        raise NameError("Something went wrong loading file for system (%s)" % (chann_type))