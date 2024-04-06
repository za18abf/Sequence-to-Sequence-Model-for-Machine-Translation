import torch
import spacy
from torchtext.data.metrics import bleu_score
import sys

def translate_the_sentence(model, sentence, german, english, device, max_length=50):

    # Load the German tokenizer from Spacy
    spacy_german = spacy.load("de_core_news_sm")

    # Tokenize the input sentence. If it's already a list of tokens, just lowercase them.
    if type(sentence) == str:
        tokens = [token.text.lower() for token in spacy_german(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    # Add start and end tokens to the sentence
    tokens.insert(0, german.init_token)
    tokens.append(german.eos_token)

    # Convert each token to its index in the vocabulary
    text_to_indices = [german.vocab.stoi[token] for token in tokens]

    # Convert the list of indices to a tensor and add a batch dimension
    sentence_tensor = torch.LongTensor(text_to_indices).unsqueeze(1).to(device)

    # Pass the source sentence through the encoder
    with torch.no_grad():
        outputs_encoder, hiddens, cells = model.encoder(sentence_tensor)

    outputs = [english.vocab.stoi["<sos>"]]

    # Use the encoder's final hidden and cell states to initialize the decoder
    for _ in range(max_length):
        previous_word = torch.LongTensor([outputs[-1]]).to(device)

        # Forward pass through the decoder
        with torch.no_grad():
            output, hiddens, cells = model.decoder(previous_word, outputs_encoder, hiddens, cells)
            best_guess = output.argmax(1).item()

        outputs.append(best_guess)

        # If the decoder predicts the <eos> token, stop decoding
        if output.argmax(1).item() == english.vocab.stoi["<eos>"]:
            break

    # Convert the indices of the output sentence to words
    translated_sentences = [english.vocab.itos[idx] for idx in outputs]

    # Return the translated sentence without the <sos> token
    return translated_sentences[1:]

def bleu(data, model, german, english, device):

    """Calculate BLEU score for a seq2seq model on a dataset."""

    targets = []
    outputs = []

    # Iterate over the dataset and generate translations
    for example in data:
        src = vars(example)["src"]
        trg = vars(example)["trg"]

        prediction = translate_the_sentence(model, src, german, english, device)
        prediction = prediction[:-1]  # remove <eos> token

        targets.append([trg])
        outputs.append(prediction)

    # Compute and return the BLEU score
    return bleu_score(outputs, targets)

def save_checkpoint(state, filename="my_checkpoint2.pth.tar"):

    """Save model and optimizer states to a file."""

    print("=>----Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):

    """Load model and optimizer states to a file."""

    print("=>----Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])