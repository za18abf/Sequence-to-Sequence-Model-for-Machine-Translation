    #----Initial phase - Importing the necessary libraries and modules----#

import torch # Torch is the main Pytorch module
import torch.nn as nn # nn is PyTorch's neural networks library
import torch.optim as optim # optim contains optimization algorithms (like SGD, Adam, etc.) for training models in PyTorch
# Multi30k is a dataset for machine translation tasks
# It contains English, German, and French sentences
from torchtext.legacy.datasets import Multi30k
# Field and BucketIterator are utility classes from torchtext
from torchtext.legacy.data import Field, BucketIterator
import numpy as np # numpy is a general-purpose array-processing library
import spacy # spacy is a library for natural language processing (NLP)
import random # random module to generate random numbers, used for randomness in training
# SummaryWriter is from the tensorboard module in PyTorch,
# It allows logging and visualizing data in TensorBoard, a visualization tool
from torch.utils.tensorboard import SummaryWriter
# utils1 seems to be a custom module (not part of standard libraries)
from utils1 import translate_the_sentence, bleu, save_checkpoint, load_checkpoint

    #----Second phase - Defining Spacy and tokenization functions for vocabulary generation----#

# Loads the German and English language models from spaCy.
spacy_german = spacy.load("de_core_news_sm")
spacy_english = spacy.load("en_core_web_sm")

# Define a function to tokenize German text using spaCy's German tokenizer.
def tokenize_german(text):
    return [tok.text for tok in spacy_german.tokenizer(text)]

# Define a function to tokenize English text using spaCy's English tokenizer.
def tokenize_english(text):
    return [tok.text for tok in spacy_english.tokenizer(text)]

# Define a Field for the German language,a Field specifies how the data should be processed.
german = Field(tokenize=tokenize_german, lower=True, init_token="<sos>", eos_token="<eos>")

# Define a Field for the English language
english = Field(tokenize=tokenize_english, lower=True, init_token="<sos>", eos_token="<eos>")

# Load the Multi30k dataset,the data is split into training, validation, and test sets.
#".de" is the extension for German sentences and ".en" is the extension for English sentences
train_data, valid_data, test_data = Multi30k.splits(exts=(".de", ".en"), fields=(german, english))

# Build vocabularies for the German and English languages based on the training data.
german.build_vocab(train_data, max_size=10000, min_freq=2)
english.build_vocab(train_data, max_size=10000, min_freq=2)

            #----Third phase - Defining Encoder and Decoder functions to support the seq2seq function----#

# Define the Encoder class, which inherits from nn.Module
class Encoder(nn.Module):

    # Constructor for the Encoder class
    def __init__(self, input_size, embedding_size, hidden_size, number_layers, p):

        # Call the constructor of the parent class (nn.Module)
        super(Encoder, self).__init__()

        # Define dropout for regularization, which helps prevent overfitting
        self.dropout = nn.Dropout(p)

        # Hidden size and number of layers are set as class attributes for easy access
        self.hidden_size = hidden_size
        self.number_layers = number_layers

        # Define an embedding layer to convert token indices to embeddings
        self.embedding = nn.Embedding(input_size, embedding_size)

        # Define a bidirectional LSTM layer
        self.rnn = nn.LSTM(embedding_size, hidden_size, number_layers, bidirectional=True)

        # Define linear layers to transform the concatenated forward and backward hidden states and cell states
        self.fc_hidden = nn.Linear(hidden_size * 2, hidden_size)
        self.fc_cell = nn.Linear(hidden_size * 2, hidden_size)

    # Define the forward pass of the Encoder
    def forward(self, x):

        # Convert token indices to embeddings using the embedding layer and apply dropout
        embedding = self.dropout(self.embedding(x))

        # Pass the embeddings through the LSTM
        encoder_states, (hidden, cell) = self.rnn(embedding)

        # Concatenate the forward and backward hidden and cell states from the LSTM
        # Then pass them through linear layers to ensure they have the correct size for the decoder
        hidden = self.fc_hidden(torch.cat((hidden[0:1], hidden[1:2]), dim=2))
        cell = self.fc_cell(torch.cat((cell[0:1], cell[1:2]), dim=2))

        # Return the encoder_states, hidden, and cell states to be used by the decoder
        return encoder_states, hidden, cell

# Define the Decoder class, which inherits from nn.Module
class Decoder(nn.Module):

    # Constructor for the Decoder class
    def __init__(self, input_size, embedding_size, hidden_size, output_size, number_layers, p):

        # Call the constructor of the parent class (nn.Module)
        super(Decoder, self).__init__()

        # Define dropout for regularization
        self.dropout = nn.Dropout(p)

        # Hidden size and number of layers are set as class attributes for easy access
        self.hidden_size = hidden_size
        self.number_layers = number_layers

        # Define an embedding layer to convert token indices to embeddings
        self.embedding = nn.Embedding(input_size, embedding_size)

        # Define an LSTM layer. Note that its input will be the sum of the embedding and the context vector, hence "hidden_size * 2 + embedding_size"
        self.rnn = nn.LSTM(hidden_size * 2 + embedding_size, hidden_size, number_layers)

        # Define a linear layer to calculate the "energy" for the attention mechanism.
        self.energy = nn.Linear(hidden_size * 3, 1)

        # Define a linear layer to transform the LSTM outputs to the desired output size
        self.fc = nn.Linear(hidden_size, output_size)

        # Softmax to compute attention weights from energies
        self.softmax = nn.Softmax(dim=0)

        # ReLU activation function, commonly used in neural networks
        self.relu = nn.ReLU()

    # Define the forward pass of the Decoder
    def forward(self, x, encoder_states, hidden, cell):

        # x is the input token for this timestep
        x = x.unsqueeze(0)

        # Convert token indices to embeddings and apply dropout
        embedding = self.dropout(self.embedding(x))

        # Compute attention weights based on the current hidden state and all encoder states
        sequence_length = encoder_states.shape[0]
        h_reshaped = hidden.repeat(sequence_length, 1, 1)
        energy = self.relu(self.energy(torch.cat((h_reshaped, encoder_states), dim=2)))
        attention = self.softmax(energy)

        # Compute the context vector as a weighted sum of encoder states, based on attention weights
        context_vector = torch.einsum("snk,snl->knl", attention, encoder_states)

        # Concatenate context vector and embedding, and use this as input to the LSTM
        rnn_input = torch.cat((context_vector, embedding), dim=2)
        outputs, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))

        # Transform LSTM outputs to target vocabulary space
        predictions = self.fc(outputs).squeeze(0)

        # Return the predictions, hidden, and cell states to be used by the decoder
        return predictions, hidden, cell

            #----Fourth phase - Defining Seq2seq function that combines encoder and decoder functions----#

class Seq2SeqAttention(nn.Module):

    # Initialize the encoder and decoder networks
    def __init__(self, encoder, decoder):
        super(Seq2SeqAttention, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target, teacher_force_ratio=0.5):

        # Define dimensions and sizes
        batch_size = source.shape[1]
        target_len = target.shape[0]
        target_vocab_size = len(english.vocab)

        # Initialize output tensor with zeros
        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(device)

        # Pass the source sequence through the encoder
        encoder_states, hidden, cell = self.encoder(source)

        # First input will be <SOS> token
        x = target[0]

        for t in range(1, target_len):

            # Decode the input token and get the next output, hidden and cell states
            output, hidden, cell = self.decoder(x, encoder_states, hidden, cell)

            # Store prediction for current time step
            outputs[t] = output

            # Get the best word the Decoder predicted (index in the vocabulary)
            best_guess = output.argmax(1)

            # Decide if we will use teacher forcing or not
            # If yes, use the actual next token from the target sequence
            # If not, use the token predicted by the decoder as the next input
            x = target[t] if random.random() < teacher_force_ratio else best_guess

        return outputs

         #----Fifth phase - Defining the hyperparameters for training and evaluation----#

# Training hyperparameters
num_epochs = 100
learning_rate = 0.003
batch_size = 34

# Model hyperparameters

# Flag to determine if a pre-trained model should be loaded
load_model = True

# Set device to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Vocabulary sizes for the encoder (German) and decoder (English)
input_size_encoder = len(german.vocab)
input_size_decoder = len(english.vocab)
output_size = len(english.vocab)

# Define the size of embeddings for the encoder and decoder
embedding_size = 300

# Define the size of hidden layers. It must be consistent across both RNNs
hidden_size = 1024  # Needs to be the same for both RNN's

# Number of layers in the RNNs
number_layers = 1

# Dropout rates for regularization in the encoder and decoder
dropout = 0.5

# Tensorboard configuration for visualization of loss during training
writer = SummaryWriter(f"EN-DE/loss_plot")
step = 0

# Create data iterators to fetch batches of data. The batches are sorted by the source sequence length for efficiency.
train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=batch_size,
    sort_within_batch=True,
    sort_key=lambda x: len(x.src),
    device=device,
)

# Instantiate the encoder and decoder networks
encoder_network = Encoder(input_size_encoder, embedding_size, hidden_size, number_layers, dropout).to(device)
decoder_network = Decoder(input_size_decoder, embedding_size, hidden_size, output_size,number_layers,dropout,).to(device)

# Combine encoder and decoder into a single Seq2SeqAttention model
model = Seq2SeqAttention(encoder_network, decoder_network).to(device)

# Define the optimizer for training. In this case, the Adam optimizer is used.
Adam_optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Get the index of the padding token from the target vocabulary (English)
pad_idx = english.vocab.stoi["<pad>"]

# Define the loss function, ignoring the padding token index for calculation
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

# If the load_model flag is set, load the pre-trained model and optimizer states
if load_model:
    load_checkpoint(torch.load("my_checkpoint2.pth.tar"), model, Adam_optimizer)

model.eval()  # Set the model to evaluation mode
                #uncomment the following code up until sys.exit to test the model and check the BLEU score
# custom_sentence = "type the german sentence to train"
# translated_sentences = translate_the_sentence(model, custom_sentence, german, english, device, max_length=50)
# print(f"Translated sentence: {translated_sentences}")
#
# score = bleu(test_data, model, german, english, device)
# print(f"Bleu score {score * 100:.2f}")
#
# import sys
# sys.exit()

                #----Sixth phase - training the model in a loop with epochs and saving the loss plot using tensorboard----#

# Define the German sentence to be translated at each epoch
training_sentence = ("Type a german sentence to train the model")

# Start the training loop
for epoch in range(num_epochs):
    print('---------------------------------------------------------')
    print(f"[Epoch {epoch + 1} / {num_epochs}]")

    # Print the German input sentence for reference
    print('Input German sentence:', training_sentence)

    # Create a checkpoint to save model's state and optimizer's state
    checkpoint = {"state_dict": model.state_dict(),"optimizer": Adam_optimizer.state_dict(),}

    # Save the current model and optimizer states
    save_checkpoint(checkpoint)

    # Set the model in evaluation mode for translation
    model.eval()

    # Translate the German sentence to English
    translated_sentences = translate_the_sentence(model, training_sentence, german, english, device, max_length=50)

    # Print the translated sentence
    print(f"Translated example sentence: \n {translated_sentences}")

    # Set the model in Training mode for translation
    model.train()

    # Iterate through the training data using the train iterator
    for batch_idx, batch in enumerate(train_iterator):
        # Get the source (German) and target (English) sequences from the batch and send them to the device
        inp_data = batch.src.to(device)
        target = batch.trg.to(device)

        # Forward pass: Pass the input data through the model to get the output
        output = model(inp_data, target)

        # Remove the start token from the output and target sequences and reshape them
        output = output[1:].reshape(-1, output.shape[2])
        target = target[1:].reshape(-1)

        # Zero out the gradients to ensure they don't accumulate
        Adam_optimizer.zero_grad()

        # Calculate the loss between the model's output and the actual target sequence
        loss = criterion(output, target)

        # Backward pass: Compute the gradient of the loss with respect to model parameters
        loss.backward()

        # Clip the gradients to ensure they stay within a safe range and prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

        # Update the model's parameters using the optimizer
        Adam_optimizer.step()

        # Log the loss to Tensorboard for visualization
        plot_graph = writer.add_scalar("Training loss", loss, global_step=step)
        step += 1