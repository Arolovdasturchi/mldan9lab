# ========================================================
# 1. Zarur kutubxonalar
# ========================================================
import numpy as np
import matplotlib
matplotlib.use("Agg")      # Grafik oynasiz, faqat faylga saqlash uchun
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, LSTM, Dense


# ========================================================
# 2. TXT DATASETNI O‘QISH
# ========================================================
with open("futbol_data_set_savol_javob.txt", "r", encoding="utf-8") as f:
    text = f.read()

tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])

total_words = len(tokenizer.word_index) + 1
print("Umumiy so‘zlar soni:", total_words)

# ========================================================
# 3. Ketma-ketliklar yaratish
# ========================================================
input_sequences = []

for line in text.split("\n"):
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram = token_list[:i+1]
        input_sequences.append(n_gram)

max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(
    pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre')
)

X = input_sequences[:, :-1]
y = input_sequences[:, -1]

print("X shape:", X.shape)
print("y shape:", y.shape)


# ========================================================
# 4. RNN MODELI
# ========================================================
model_rnn = Sequential()
model_rnn.add(Embedding(total_words, 64, input_length=max_sequence_len - 1))
model_rnn.add(SimpleRNN(128, activation='tanh'))
model_rnn.add(Dense(total_words, activation='softmax'))

model_rnn.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

print("\nRNN modeli o‘qitilmoqda...\n")
history_rnn = model_rnn.fit(X, y, epochs=5, verbose=1)  # <- EPOCHS 5 ga qisqartirildi


# ========================================================
# 5. LSTM MODELI
# ========================================================
model_lstm = Sequential()
model_lstm.add(Embedding(total_words, 64, input_length=max_sequence_len - 1))
model_lstm.add(LSTM(128, activation='tanh'))
model_lstm.add(Dense(total_words, activation='softmax'))

model_lstm.compile(loss='sparse_categorical_crossentropy',
                   optimizer='adam',
                   metrics=['accuracy'])

print("\nLSTM modeli o‘qitilmoqda...\n")
history_lstm = model_lstm.fit(X, y, epochs=5, verbose=1)  # <- EPOCHS 5 ga qisqartirildi


# ========================================================
# 6. NATIJALAR GRAFIKINI FAYLGA SAQLASH
# ========================================================
plt.plot(history_rnn.history['accuracy'], label="RNN Accuracy")
plt.plot(history_lstm.history['accuracy'], label="LSTM Accuracy")
plt.title("RNN va LSTM Model Natijalari")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

# Grafikni saqlash
plt.savefig("model_results.png")
print("\nGrafik 'model_results.png' faylga saqlandi.")
