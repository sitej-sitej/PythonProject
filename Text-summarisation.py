import torch
import torch.nn as nn
import torch.nn.functional as F
import nltk
from transformers import BartTokenizer, BartForConditionalGeneration
from nltk.tokenize import sent_tokenize, word_tokenize

nltk.download("punkt")

class CNNEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, kernel_sizes, num_filters):
        super(CNNEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (k, embedding_dim)) for k in kernel_sizes
        ])

    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(item, item.size(2)).squeeze(2) for item in x]
        out = torch.cat(x, 1)
        return out

class ExtractiveSummarizer(nn.Module):
    def __init__(self, encoder, hidden_dim):
        super(ExtractiveSummarizer, self).__init__()
        self.encoder = encoder
        self.fc = nn.Linear(encoder.convs[0].out_channels * len(encoder.convs), hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        features = self.encoder(x)
        hidden = F.relu(self.fc(features))
        out = torch.sigmoid(self.classifier(hidden))
        return out

def build_vocab(sentences):
    vocab = {"<PAD>": 0}
    idx = 1
    for sent in sentences:
        for word in word_tokenize(sent.lower()):
            if word not in vocab:
                vocab[word] = idx
                idx += 1
    return vocab

def encode_sentence(sentence, vocab, max_len):
    tokens = word_tokenize(sentence.lower())
    ids = [vocab.get(word, 0) for word in tokens[:max_len]]
    return ids + [0] * (max_len - len(ids))

if __name__ == "__main__":
    text = text = """
Climate change is one of the most pressing issues facing humanity today. Rising global temperatures have led to more frequent and intense natural disasters, such as hurricanes, floods, and wildfires. Scientists agree that human activities, particularly the burning of fossil fuels, are a major contributor to greenhouse gas emissions. As a result, many countries are investing in renewable energy sources like wind, solar, and hydroelectric power. There is also a growing movement toward sustainable practices, including reforestation, electric vehicles, and reducing single-use plastics. While challenges remain, global cooperation and innovation offer hope for a more sustainable future.
"""

    print("\n Extractive Summarization:\n")

    sentences = sent_tokenize(text)
    vocab = build_vocab(sentences)
    max_len = 15

    encoded = [encode_sentence(s, vocab, max_len) for s in sentences]
    input_tensor = torch.tensor(encoded)

    vocab_size = len(vocab)
    embedding_dim = 50
    kernel_sizes = [2, 3, 4]
    num_filters = 16
    hidden_dim = 32

    encoder = CNNEncoder(vocab_size, embedding_dim, kernel_sizes, num_filters)
    model = ExtractiveSummarizer(encoder, hidden_dim)
    with torch.no_grad():
        scores = model(input_tensor).squeeze()

    top_indices = torch.topk(scores, k=2).indices
    summary_extract = [sentences[i] for i in sorted(top_indices.tolist())]

    print("Extracted Sentences:")
    for sent in summary_extract:
        print("-", sent)

    print("\nAbstractive Summarization:\n")

    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

    inputs = tokenizer([text], max_length=1024, return_tensors="pt", truncation=True)
    summary_ids = model.generate(inputs["input_ids"], max_length=100, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)

    summary_abs = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    print(summary_abs)
