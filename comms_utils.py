import torch
import numpy as np
import math

# To do block encoding (Hamming)
import sk_dsp_comm.fec_block as block

def channel(x, n, k, snr_db, chann_type="AWGN"):
    """
    Definition of the channel. e.g. Add AWGN to the samples
    Args:
        x of shape (batch_size, k): Encoded messages
        n (int): Length of the encoded messages
        k (int): Length of the actual messages
        snr_db (float): SNR to add noise
        chann_type (string): Channel type. Currently only AWGN available
    Returns:
        x_channel of shape (batch_size, k): x with noise
    """
    # Transform from dB to linear
    snr_lin = 10**(snr_db/10)
    # Get the rate of the encoding
    rate = k/n
    
    if chann_type == "AWGN":
        # Finally calculate the variance of the AWGN
        n0 = 1/(snr_lin*rate)
        var_channel = math.sqrt(n0/2)

        # Use the reparametrization trick to apply noise to x
        if torch.is_tensor(x):
            x_channel = x + var_channel*torch.randn_like(x)
        else:
            x_channel = x + var_channel*np.random.randn(*x.shape)

        return x_channel
    else:
        raise NameError("Channel type not supported.")


def block_encoder(x, n, k):
    """
    This is going to be the definition of encoding using Hamming
    Args:
        x of shape (batch_size, k): Messages without encoding
        n (int): Length of the encoded messages
        k (int): Length of the actual messages
    Returns:
        y of shape (batch_size, n): Encoded messages with Hamming
    """
    # There is no need for encoding
    if n == k:
        # Return as float because that the way encoder.hamm_encoder returns it
        return x
    
    # We initialize the encoder with the number of parity bits that we need
    # According to doc from block.fec_hamming
    # Initialized with j. Where n = 2^j-1. k = n-j.
    encoder = block.fec_hamming(n-k)
    
    # Get the batch size and pre-allocate adequate space for it
    batch_size, _ = x.shape
    encoding_results = np.zeros((batch_size, n), dtype=int)
    
    # Iterate over the batches and get the encoding for all of them
    for i, x_vec in enumerate(x):
        encoding_results[i, :] = encoder.hamm_encoder(x_vec)
    
    return encoding_results


def block_decoder(y, n, k):
    """
    This is going to be the definition of decoding using Hamming
    Args:
        x of shape (batch_size, n): Encoded messages
        n (int): Length of the encoded messages
        k (int): Length of the actual messages
    Returns:
        y of shape (batch_size, k): Decoded messages with Hamming
    """
    # There is no need for decoding
    if n == k:
        # Return as float because that the way encoder.hamm_decoder returns it
        return y
    
    # We initialize the decoder with the number of parity bits that we need
    # According to doc from block.fec_hamming
    # Initialized with j. Where n = 2^j-1. k = n-j.
    decoder = block.fec_hamming(n-k)
    
    # Get the batch size and pre-allocate adequate space for it
    batch_size, _ = y.shape
    decoding_results = np.zeros((batch_size, k), dtype=int)
    
    # Iterate over the batches and get the encoding for all of them
    for i, y_vec in enumerate(y):
        decoding_results[i, :] = decoder.hamm_decoder(y_vec)
    
    return decoding_results


def bler(x, y):
    """
    Function to get the BLER
    Args:
        x (numpy array): Original samples
        y (numpy array): Decoded samples
    Returns:
        y of shape (batch_size, k): Decoded messages with Hamming
    """
    # Get the total number of messages
    batch_size, _ = x.shape
    
    # Check where are the errors between received and transmitted
    errors = (x != y)
    # How many errors per block
    errors_block = errors.sum(axis=1)
    # If there was an error in the block count it as bad block
    total_errors = (errors_block > 0).sum()

    return total_errors/batch_size