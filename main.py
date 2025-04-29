"""
Sentiment Analyzer AI (Inefficient Baseline – Partially Optimized)
"""

import random
import math
import json
import os
import numpy as np

# Tekrarlanabilirlik için sabit seed
random.seed(42)

class DataGenerator:
    """
    Generates synthetic text data for binary sentiment classification.
    Each sample is a random sequence of 'wordX'. Label is based on token sum.
    """
    def __init__(self, vocab_size=100, sentence_length=10, num_samples=1000, log_file="data_log.jsonl"):
        self.vocab = [f"word{i}" for i in range(vocab_size)]
        self.sentence_length = sentence_length
        self.num_samples = num_samples
        self.log_file = log_file
        if os.path.exists(self.log_file):
            os.remove(self.log_file)

    def generate(self):
        samples = []
        labels = []
        for _ in range(self.num_samples):  # #1: Python döngülü veri üretim
            words = []
            for _ in range(self.sentence_length):
                words.append(random.choice(self.vocab))
            sentence = " ".join(words)
            score = sum(int(w.replace("word", "")) for w in words)
            label = 1 if score > (len(self.vocab) * self.sentence_length / 2) else 0
            samples.append(sentence)
            labels.append(label)
            # #2: Her örnek için dosyaya aç-yaz-kapat I/O
            with open(self.log_file, "a") as f:
                f.write(json.dumps({"sentence": sentence, "label": label}) + "\n")
        return samples, labels

class SimpleNN:
    """
    A simple two-layer neural network for binary classification.
    """
    def __init__(self, input_size, hidden_size):
        # Ağırlıkları vektörel başlatıyoruz
        self.w1 = np.random.uniform(-0.5, 0.5, (hidden_size, input_size))
        self.b1 = np.random.uniform(-0.5, 0.5, hidden_size)
        self.w2 = np.random.uniform(-0.5, 0.5, hidden_size)
        self.b2 = np.random.uniform(-0.5, 0.5)

    def _sigmoid(self, x):
        # vektörleştirilmiş sigmoid
        return 1 / (1 + np.exp(-x))

    def forward(self, x_vec):
        # Gizli katman vektörel hesap
        hidden = self._sigmoid(np.dot(self.w1, x_vec) + self.b1)
        # Çıkış katmanı vektörel hesap
        return self._sigmoid(np.dot(self.w2, hidden) + self.b2)

    def train(self, samples, labels, epochs=5, lr=0.01):
        """
        Basit eğitim: sadece forward pass ve hata hesabı ile dummy döngü
        (geri yayılım yok)
        """
        for epoch in range(epochs):
            for idx, sent in enumerate(samples):
                # Bag-of-words vektörizasyonu (inefficient loop)
                x_vec = [0] * self.w1.shape[1]
                for tok in sent.split():
                    x_vec[int(tok.replace("word", "")) % len(x_vec)] += 1
                _ = self.forward(x_vec)
            # Her epoch sonunda özet satırı
            print(f"Epoch {epoch+1}/{epochs} completed.")
        return

def main():
    config = {
        "vocab_size": 500,
        "sentence_length": 20,
        "num_samples": 2000,
        "hidden_size": 64,
        "epochs": 3,
        "learning_rate": 0.05
    }

    gen = DataGenerator(
        vocab_size=config["vocab_size"],
        sentence_length=config["sentence_length"],
        num_samples=config["num_samples"]
    )
    samples, labels = gen.generate()

    nn = SimpleNN(input_size=config["vocab_size"], hidden_size=config["hidden_size"])
    nn.train(samples, labels, epochs=config["epochs"], lr=config["learning_rate"])

if __name__ == "__main__":
    main()
