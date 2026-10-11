import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping

# 1. Set Parameters
VOCAB_SIZE = 10000
MAX_LENGTH = 200

# 2. Load Dataset
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=VOCAB_SIZE)
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# 3. Pad Sequences
X_train = pad_sequences(X_train, maxlen=MAX_LENGTH)
X_test = pad_sequences(X_test, maxlen=MAX_LENGTH)

# 4. Build Model
model = Sequential([
    Embedding(VOCAB_SIZE, 128, input_length=MAX_LENGTH),
    LSTM(128, dropout=0.2, recurrent_dropout=0.2),
    Dense(64, activation="relu"),
    Dropout(0.5),
    Dense(1, activation="sigmoid")
])

# 5. Compile Model
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# 6. Train Model
early_stopping = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=5, batch_size=128,
                    validation_data=(X_test, y_test), callbacks=[early_stopping])

# 7. Evaluate Model
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {acc:.2f}")

# 8. Prediction Function
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

def encode_sentence(sentence):
    tokens = [word_index.get(word, 2) for word in sentence.lower().split()]  # 2 = OOV
    return pad_sequences([tokens], maxlen=MAX_LENGTH)

def predict_sentiment(sentence):
    encoded = encode_sentence(sentence)
    prediction = model.predict(encoded)[0][0]
    sentiment = "Positive" if prediction >= 0.5 else "Negative"
    confidence = prediction * 100 if sentiment == "Positive" else (1 - prediction) * 100
    return sentiment, confidence

# 9. Test Sentences
test_sentences = [
    "This movie was amazing and the acting was excellent.",
    "I really enjoyed this film and would recommend it to everyone.",
    "The story was beautiful and the characters were wonderful.",
    "This was one of the best movies I have ever watched.",
    "The movie was terrible and extremely boring.",
    "I hated this film because the acting was awful.",
    "This was a complete waste of time and money.",
    "The story was boring and the characters were disappointing.",
    "I would never recommend this horrible movie."
]

print("\nSentiment Predictions:")
for sentence in test_sentences:
    sentiment, confidence = predict_sentiment(sentence)
    print(f"Sentence: {sentence}")
    print(f"Sentiment: {sentiment}, Confidence: {confidence:.2f}%\n")

# 10. Interactive Sentiment Analysis
print("\nCustom Sentiment Analysis (type 'exit' to stop)")
while True:
    user_text = input("Enter a sentence: ")
    if user_text.lower() == "exit":
        print("Program ended.")
        break
    if user_text.strip() == "":
        print("Please enter a valid sentence.")
        continue
    sentiment, confidence = predict_sentiment(user_text)
    print(f"Predicted Sentiment: {sentiment}, Confidence: {confidence:.2f}%")
