"""
Sentiment Analyzer AI (Inefficient Baseline)
"""

import random
import math
import json
import os

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
        # Her çalışmada eski log’u sil
        if os.path.exists(self.log_file):
            os.remove(self.log_file)

    def generate(self):
        samples = []
        labels = []
        for _ in range(self.num_samples):  # #1: Python döngülü veri üretim
            words = []
            for _ in range(self.sentence_length):  # nested loop
                words.append(random.choice(self.vocab))
            sentence = " ".join(words)
            score = sum(int(w.replace("word", "")) for w in words)
            label = 1 if score > (len(self.vocab) * self.sentence_length / 2) else 0
            samples.append(sentence)
            labels.append(label)
            # #2: Her örnek için dosyaya yazma (yavaş I/O)
            with open(self.log_file, "a") as f:
                f.write(json.dumps({"sentence": sentence, "label": label}) + "\n")
        return samples, labels

class SimpleNN:
    """
    A simple two-layer neural network for binary classification.
    """
    def __init__(self, input_size, hidden_size):
        # Ağırlıkları rastgele başlat
        self.w1 = [[random.uniform(-0.5, 0.5) for _ in range(input_size)] for _ in range(hidden_size)]
        self.b1 = [random.uniform(-0.5, 0.5) for _ in range(hidden_size)]
        self.w2 = [random.uniform(-0.5, 0.5) for _ in range(hidden_size)]
        self.b2 = random.uniform(-0.5, 0.5)

    def _sigmoid(self, x):
        # #3: math.exp kullanan yavaş sigmoid
        return 1 / (1 + math.exp(-x))

    def forward(self, x_vec):
        # Gizli katman
        hidden = []
        for i in range(len(self.w1)):  # #4: manuel nested loops
            z = 0
            for j in range(len(x_vec)):
                z += self.w1[i][j] * x_vec[j]
            z += self.b1[i]
            hidden.append(self._sigmoid(z))
        # Çıkış katmanı
        out_z = 0
        for i, h in enumerate(hidden):
            out_z += self.w2[i] * h
        out_z += self.b2
        return self._sigmoid(out_z)

    def train(self, samples, labels, epochs=5, lr=0.01):
        """
        Basit eğitim: sadece forward pass ve hata hesabı ile dummy döngü
        (geri yayılım yok)
        """
        for epoch in range(epochs):
            for idx, sent in enumerate(samples):
                # Bag-of-words vektörizasyonu (inefficient loop)
                x_vec = [0] * len(self.w1[0])
                for tok in sent.split():  # ek bir Python döngüsü
                    x_vec[int(tok.replace("word", "")) % len(x_vec)] += 1
                _ = self.forward(x_vec)  # sadece ileri geçiş
            # Her epoch sonunda tek bir özet satırı
            print(f"Epoch {epoch+1}/{epochs} completed.")
        return

def main():
    # Konfigürasyon
    config = {
        "vocab_size": 500,
        "sentence_length": 20,
        "num_samples": 2000,
        "hidden_size": 64,
        "epochs": 3,
        "learning_rate": 0.05
    }

    # 1) Veri üret
    gen = DataGenerator(
        vocab_size=config["vocab_size"],
        sentence_length=config["sentence_length"],
        num_samples=config["num_samples"]
    )
    samples, labels = gen.generate()

    # 2) Modeli oluştur ve eğit
    nn = SimpleNN(input_size=config["vocab_size"], hidden_size=config["hidden_size"])
    nn.train(samples, labels, epochs=config["epochs"], lr=config["learning_rate"])

if __name__ == "__main__":
    main()
