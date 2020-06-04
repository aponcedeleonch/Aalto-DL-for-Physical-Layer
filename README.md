# Aalto University - Comparing DL approaches on Physical Layer. *An Introduction to Deep Learning for the Physical Layer* vs *End-to-End Learning of Communications Systems Without a Channel Model*

Developed at Aalto University as part of the course Learning to Communicate Optimally by:
* Alejandro Ponce de León Chávez

The purpose of this project is to compare two Deep Learning approaches to learn proper encodings in a traditional communications system i.e. Transmitter -> Channel -> Receiver. The first approach can be found in [An Introduction to Deep Learning for the Physical Layer](https://arxiv.org/pdf/1702.00832.pdf) by O'Shea and Hoydis, 2017. The second one is [End-to-End Learning of Communications Systems Without a Channel Model](https://arxiv.org/pdf/1804.02276.pdf) by Aoudia and Hoydis, 2018. The difference between them is that the work by O'Shea and Hoydis relies that we know the gradients of the channel while the one by Aoudia and Hoydis doesn't. Instead Aoudia and Hoydis approach has a feedback from the receiver to the transmitter and uses a policy converting it in a Reinforcement Learning task.

## Organization of the repository

- `introducion_phis_layer_autoencoder.ipynb` – Notebook focused on implementing paper “An Introduction to Deep Learning for the Physical Layer”  by O’Shea and Hoydis
- `end_to_end_no_channel.ipynb` – Notebook focused on implementing “End-to-End Learning of Communications Systems Without a Channel Model” by Auodia and Hoydis
comparison_channel_nochannel.ipynb – Notebook focused on comparing last two approaches
- `models.py` – Pytorch classes where the models for both approaches are defined
- `utils.py` – Miscellaneous functions used for e.g. plotting
- `comms_utils.py` – Miscellaneous used for simulating communication with BPSK and Hamming encoding
- `trained_models/*` - Pre-trained Pytorch models
