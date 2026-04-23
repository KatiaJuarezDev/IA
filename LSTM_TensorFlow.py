from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

texto = "El gato está durmiendo en el sofá. " * 100
caracteres = sorted(list(set(texto)))
char_to_idx = {char: idx for idx, char in enumerate(caracteres)}
idx_to_char = {idx: char for char, idx in char_to_idx.items()}

longitud_secuencia = 40
X, y = [], []
for i in range(len(texto) - longitud_secuencia):
    secuencia = texto[i:i+longitud_secuencia]
    siguiente = texto[i+longitud_secuencia]
    X.append([char_to_idx[char] for char in secuencia])
    y.append(char_to_idx[siguiente])

X = np.array(X)
y = keras.utils.to_categorical(y, num_classes=len(caracteres))

modelo = keras.Sequential([
    layers.Input(shape=(longitud_secuencia,), dtype="int32"),
    layers.Embedding(len(caracteres), 50),
    layers.LSTM(128, return_sequences=True),
    layers.LSTM(128),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(caracteres), activation='softmax')
])

modelo.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

modelo.fit(X, y, epochs=50, batch_size=128)

def generar_texto(semilla, longitud=100):
    texto_generado = semilla
    for _ in range(longitud):
        x = np.array([[char_to_idx[char] for char in texto_generado[-longitud_secuencia:]]])
        pred = modelo.predict(x, verbose=0)
        siguiente_idx = np.argmax(pred[0])
        siguiente_char = idx_to_char[siguiente_idx]
        texto_generado += siguiente_char
    return texto_generado

print(generar_texto("El gato está ", 50))