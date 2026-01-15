# ANN CODES

```python
# 1️⃣ Import Libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# 2️⃣ Build the ANN Model
ann = Sequential()

# Hidden Layer 1
ann.add(Dense(
    units=11,          # 🔹 Number of neurons in this layer
    activation='relu', # 🔹 Activation function (ReLU = Rectified Linear Unit)
    input_dim=X_train.shape[1]  # 🔹 Number of input features (only for first layer)
))
ann.add(Dropout(0.2))  # 🔹 Dropout layer, 20% of neurons are randomly ignored during training

# Hidden Layer 2
ann.add(Dense(
    units=7,
    activation='relu'
))
ann.add(Dropout(0.2))  # 🔹 Helps prevent overfitting by randomly dropping neurons

# Hidden Layer 3
ann.add(Dense(
    units=7,
    activation='relu'
))
ann.add(Dropout(0.2))

# Output Layer
ann.add(Dense(
    units=1,          # 🔹 1 neuron for binary classification
    activation='sigmoid'  # 🔹 Sigmoid outputs probability between 0 and 1
))

# 3️⃣ Compile the Model
ann.compile(
    optimizer='adam',             # 🔹 Optimizer: adjusts weights to minimize loss
    loss='binary_crossentropy',   # 🔹 Loss function: measures error for binary classification
    metrics=['accuracy']          # 🔹 Metric to evaluate performance
)

# 4️⃣ EarlyStopping to prevent overfitting
earlystopping = EarlyStopping(
    monitor='val_loss',    # 🔹 Metric to monitor (validation loss)
    min_delta=0.0001,      # 🔹 Minimum change to qualify as improvement
    patience=20,           # 🔹 Number of epochs with no improvement after which training stops
    verbose=1,             # 🔹 Print message when stopping
    mode='auto',           # 🔹 'min' or 'max' mode auto-selected
    restore_best_weights=True  # 🔹 After stopping, restore model weights from best epoch
)

# 5️⃣ Train the Model
history = ann.fit(
    X_train, y_train,          # 🔹 Training data
    validation_split=0.33,     # 🔹 33% of training data used for validation
    batch_size=10,             # 🔹 Number of samples per gradient update
    epochs=1000,               # 🔹 Maximum number of epochs (iterations over entire dataset)
    callbacks=[earlystopping]  # 🔹 Apply early stopping
)

model.summary() ## Displays the complete neural network architecture, including layers, output shapes, and trainable parameters.
y_pred_prob = model.predict(X_test_tfidf) ### Generates probability predictions (values between 0 and 1) for the test data.
y_pred = (y_pred_prob > 0.5).astype(int) #### Converts predicted probabilities into binary class labels using a 0.5 threshold.

print("Accuracy:", accuracy_score(y_test, y_pred)) ### Calculates and prints the overall prediction accuracy of the model.
print("Classification Report:\n", classification_report(y_test, y_pred)) #### Prints precision, recall, F1-score, and support for each class.

model.save('fake-news-detector.h5')  #### Saves the trained model to disk for future loading and inference.
```
```
""" What is an Epoch?

An epoch is one complete pass of the entire training dataset through the neural network.

Think of it like this:

Suppose your dataset has 1000 samples.

Your batch size is 10.

That means it will take 100 batches to complete 1 epoch (1000 ÷ 10 = 100).

So after the model has seen all 1000 samples once, that’s 1 epoch. """
```

# RNN
## ANN → No memory, works on independent features
## RNN → Has memory, works on sequential / time-dependent data

```python
# 1️⃣ Import Libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout, Embedding
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, classification_report

# 2️⃣ Build the RNN Model
rnn = Sequential()

# Embedding Layer
rnn.add(Embedding(
    input_dim=10000,      # 🔹 Vocabulary size (number of unique words)
    output_dim=128,       # 🔹 Dimension of word embedding vectors
    input_length=100      # 🔹 Length of each input sequence
))
# 🔹 Converts integer-encoded text into dense vector representations

# RNN Layer
rnn.add(SimpleRNN(
    units=64,             # 🔹 Number of RNN neurons
    activation='tanh',    # 🔹 Activation function for RNN cells
    return_sequences=False # 🔹 False since this is the last RNN layer
))
# 🔹 Learns sequential and temporal dependencies in data

# Dropout Layer
rnn.add(Dropout(0.2))
# 🔹 Randomly disables 20% neurons during training to reduce overfitting

# Output Layer
rnn.add(Dense(
    units=1,              # 🔹 Single neuron for binary classification
    activation='sigmoid'  # 🔹 Outputs probability between 0 and 1
))

# 3️⃣ Compile the Model
rnn.compile(
    optimizer='adam',              # 🔹 Optimizer to update weights
    loss='binary_crossentropy',    # 🔹 Loss function for binary classification
    metrics=['accuracy']           # 🔹 Performance evaluation metric
)

# 4️⃣ EarlyStopping Callback
earlystopping = EarlyStopping(
    monitor='val_loss',     # 🔹 Monitor validation loss
    min_delta=0.0001,       # 🔹 Minimum improvement threshold
    patience=10,            # 🔹 Stop training after 10 epochs with no improvement
    verbose=1,              # 🔹 Print stopping message
    restore_best_weights=True # 🔹 Restore best-performing model weights
)

# 5️⃣ Train the Model
history = rnn.fit(
    X_train, y_train,        # 🔹 Sequential training data
    validation_split=0.2,    # 🔹 20% data used for validation
    batch_size=32,           # 🔹 Samples per gradient update
    epochs=100,              # 🔹 Maximum number of epochs
    callbacks=[earlystopping] # 🔹 Apply early stopping
)

# 6️⃣ Model Evaluation
rnn.summary()
# 🔹 Displays model architecture, layers, and trainable parameters

y_pred_prob = rnn.predict(X_test)
# 🔹 Generates probability predictions for test data

y_pred = (y_pred_prob > 0.5).astype(int)
# 🔹 Converts probabilities into binary class labels using 0.5 threshold

print("Accuracy:", accuracy_score(y_test, y_pred))
# 🔹 Prints overall accuracy of the model

print("Classification Report:\n", classification_report(y_test, y_pred))
# 🔹 Prints precision, recall, F1-score, and support

# 7️⃣ Save the Model
rnn.save('basic-rnn-model.h5')
# 🔹 Saves trained RNN model for future loading and inference
```

# CNN used for images

```python
import tensorflow as tf
from tensorflow.keras import layers, models, mixed_precision
import os

# 1. ACTIVATE MIXED PRECISION (Boosts GPU speed significantly)
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# -------------------------------
# PARAMETERS
# -------------------------------
IMG_SIZE = 224
BATCH_SIZE = 16  # Increased from 16 for better GPU utilization
NUM_CLASSES = 7
train_dir = "/root/.cache/kagglehub/datasets/fahadullaha/facial-emotion-recognition-dataset/versions/1/processed_data"

# -------------------------------
# CLASS WEIGHTS (NO RAM OVERLOAD)
# -------------------------------
class_names = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
class_counts = {c: len(os.listdir(os.path.join(train_dir, c))) for c in class_names}

total_samples = sum(class_counts.values())
class_weights = {i: total_samples / (NUM_CLASSES * count) for i, count in enumerate(class_counts.values())}
print("Class counts:", class_counts)
print("Class weights:", class_weights)

# -------------------------------
# DATA PIPELINE (The "Fast" Way)
# -------------------------------
# This replaces ImageDataGenerator
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode='categorical' 
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)

# Optimization: Cache and Prefetch
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.shuffle(100).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# -------------------------------
# DATA AUGMENTATION (Built into the model)
# -------------------------------
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# -------------------------------
# MODEL
# -------------------------------
base_model = tf.keras.applications.MobileNetV2(
    weights='imagenet', 
    include_top=False, 
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base_model.trainable = False

inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = data_augmentation(inputs)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.3)(x)

# Important: Use float32 for the final softmax when using Mixed Precision
outputs = layers.Dense(NUM_CLASSES, activation='softmax', dtype='float32')(x)

model = models.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
)

# -------------------------------
# TRAIN
# -------------------------------

earlystopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=4, 
    restore_best_weights=True
)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=30,
    callbacks=[earlystopping],
    class_weight=class_weights
)

model.save('emotional_mobilenet_v2.keras')
```
```python
# Import TensorFlow for building and training deep learning models
import tensorflow as tf

# Import NumPy for numerical operations
import numpy as np

# Import EarlyStopping callback to stop training when validation performance stops improving
from tensorflow.keras.callbacks import EarlyStopping

# Import Sequential model to stack layers line-by-line
from tensorflow.keras.models import Sequential

# Import CNN-related layers
from tensorflow.keras.layers import (
    Conv2D,        # Performs convolution to extract spatial features
    MaxPooling2D,  # Reduces spatial dimensions of feature maps
    Dense,         # Fully connected neural network layer
    Flatten,       # Converts 2D feature maps into 1D feature vector
    Dropout,       # Randomly disables neurons to reduce overfitting
    Rescaling      # Scales pixel values to a smaller range
)

# Enable debug mode for tf.data pipelines (helps identify dataset-related issues)
tf.data.experimental.enable_debug_mode()

# Data augmentation model to increase dataset diversity
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),   # Randomly flip images horizontally
    tf.keras.layers.RandomRotation(0.2),        # Randomly rotate images
    tf.keras.layers.RandomZoom(0.2),            # Randomly zoom images
    tf.keras.layers.RandomContrast(0.2),        # Randomly adjust image contrast
])

# Define image dimensions for model input
IMG_SIZE = (128, 128)

# Define number of samples processed per batch
BATCH_SIZE = 32

# Seed value for reproducible dataset splitting
SEED = 42

# Total number of training epochs
EPOCHS = 50

# Load training dataset from directory with automatic labels
train_ds = tf.keras.utils.image_dataset_from_directory(
    r"/kaggle/input/kaggle-cat-vs-dog-dataset/kagglecatsanddogs_3367a/PetImages",  # Dataset path
    validation_split=0.2,        # Use 20% data for validation
    subset="training",           # Specify training subset
    seed=SEED,                   # Seed for consistent split
    image_size=IMG_SIZE,         # Resize images
    batch_size=BATCH_SIZE,       # Batch size
    color_mode="rgb",            # Load images in RGB format
    interpolation="bilinear"     # Resize interpolation method
)

# Load validation dataset from same directory
val_ds = tf.keras.utils.image_dataset_from_directory(
    r"/kaggle/input/kaggle-cat-vs-dog-dataset/kagglecatsanddogs_3367a/PetImages",  # Dataset path
    validation_split=0.2,        # Use same validation split
    subset="validation",         # Specify validation subset
    seed=SEED,                   # Same seed for consistency
    image_size=IMG_SIZE,         # Resize images
    batch_size=BATCH_SIZE,       # Batch size
    color_mode="rgb",            # RGB images
    interpolation="bilinear"     # Resize interpolation
)

# Print detected class labels
print("Classes:", train_ds.class_names)  # ['Cat', 'Dog']

# Ignore corrupted or unreadable images in dataset
train_ds = train_ds.apply(tf.data.experimental.ignore_errors())
val_ds   = val_ds.apply(tf.data.experimental.ignore_errors())

# Create normalization layer to scale pixel values between 0 and 1
normalization_layer = tf.keras.layers.Rescaling(1./255)

# Apply normalization to training data
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))

# Apply normalization to validation data
val_ds   = val_ds.map(lambda x, y: (normalization_layer(x), y))

# Automatically tune dataset performance
AUTOTUNE = tf.data.AUTOTUNE

# Shuffle, repeat, cache, and prefetch training dataset for performance
train_ds = train_ds.shuffle(1000).repeat().cache().prefetch(AUTOTUNE)

# Cache and prefetch validation dataset
val_ds = val_ds.cache().prefetch(AUTOTUNE)

# Build CNN model using Sequential API
cnn = Sequential([
    tf.keras.layers.Input(shape=(128,128,3)),  # Define input shape for images
    data_augmentation,                         # Apply data augmentation only during training

    Conv2D(32, (3,3), activation='relu', padding='same'),  # First convolution layer
    MaxPooling2D(),                              # Reduce spatial size

    Conv2D(64, (3,3), activation='relu', padding='same'),  # Second convolution layer
    MaxPooling2D(),                              # Downsampling

    Conv2D(128, (3,3), activation='relu', padding='same'), # Third convolution layer
    MaxPooling2D(),                              # Downsampling

    Conv2D(256, (3,3), activation='relu', padding='same'), # Fourth convolution layer
    MaxPooling2D(),                              # Downsampling

    Flatten(),                                  # Convert feature maps to 1D vector
    Dense(256, activation='relu'),              # Fully connected layer
    Dropout(0.5),                               # Drop 50% neurons to prevent overfitting
    Dense(1, activation='sigmoid')              # Output layer for binary classification
])

# Compile the CNN model
cnn.compile(
    optimizer='adam',                   # Adaptive optimizer
    loss='binary_crossentropy',         # Loss function for binary classification
    metrics=['accuracy']                # Metric to evaluate model performance
)

# Display CNN architecture summary
cnn.summary()

# Define EarlyStopping to stop training when validation loss stops improving
early_stop = EarlyStopping(
    monitor='val_loss',                 # Monitor validation loss
    patience=3,                         # Stop after 3 epochs with no improvement
    restore_best_weights=True           # Restore best model weights
)

# Train the CNN model
history = cnn.fit(
    train_ds,                           # Training dataset
    validation_data=val_ds,             # Validation dataset
    epochs=EPOCHS,                      # Maximum number of epochs
    callbacks=[early_stop]              # Apply early stopping
)

# Save the trained CNN model to disk
cnn.save('dog-vs-cat-classifier5.h5')

# Confirm successful save
print("File saved")
```

```python
import tensorflow as tf
import numpy as np
import os
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dense, Dropout,
    Flatten, BatchNormalization
)
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# =========================
# CONFIGURATION
# =========================
IMG_SIZE = (160, 160)
BATCH_SIZE = 32
EPOCHS = 20
MODEL_PATH = "plant_disease_model.keras"

# =========================
# DATA GENERATORS
# =========================
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

train_gen = datagen.flow_from_directory(
    r"/content/data/plant-data/Image Data base/Image Data base",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True
)

val_gen = datagen.flow_from_directory(
    r"/content/data/plant-data/Image Data base/Image Data base",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

NUM_CLASSES = train_gen.num_classes
CLASS_NAMES = list(train_gen.class_indices.keys())

print("Classes:", CLASS_NAMES)

# =========================
# MODEL DEFINITION
# =========================
model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(224,224,3)),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation="relu"),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation="relu"),
    BatchNormalization(),
    MaxPooling2D(2,2),

    GlobalAveragePooling2D(),
    Dense(256, activation="relu"),
    Dropout(0.5),

    Dense(NUM_CLASSES, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =========================
# CALLBACKS
# =========================
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    MODEL_PATH,
    monitor="val_accuracy",
    save_best_only=True
)

labels = train_gen.classes
class_weights_list = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(labels),
    y=labels
)

class_weights = dict(enumerate(class_weights_list))
print("Class weights:", class_weights)
for k in class_weights:
    class_weights[k] = min(class_weights[k], 20.0)
# =========================
# TRAIN MODEL
# =========================
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=[early_stop, checkpoint]
)

# =========================
# SAVE FINAL MODEL
# =========================
model.save('Plant_disease.keras')
print("Model saved")
```

# NLP 
```python
# Import NLTK library for Natural Language Processing
import nltk

# Sample text corpus
corpus = "Hello Imran. I am your friend."

# Import different tokenizers
from nltk.tokenize import sent_tokenize, word_tokenize, wordpunct_tokenize
from nltk.tokenize import TreebankWordDetokenizer

# Another sample text
corp2 = "Imran's house"

# Sentence tokenization: paragraph → sentences
print(sent_tokenize(corpus))
# OUTPUT: ['Hello Imran.', 'I am your friend.']

# Word tokenization: paragraph → words
print(word_tokenize(corpus))
# OUTPUT: ['Hello', 'Imran', '.', 'I', 'am', 'your', 'friend', '.']

# Word punctuation tokenizer: treats punctuation separately
print(wordpunct_tokenize(corp2))
# OUTPUT: ['Imran', "'", 's', 'house']

# Detokenizer: joins words back into sentence without separating punctuation
print(TreebankWordDetokenizer().tokenize(word_tokenize(corpus)))
# OUTPUT: Hello Imran. I am your friend.

# List of words for stemming demonstration
words = ['eating', 'eats', 'eaten', 'writing', 'writes', 'programming', 'study']

# Import Porter Stemmer
from nltk.stem import PorterStemmer

# Initialize Porter Stemmer
stemming = PorterStemmer()

# Apply stemming to each word
for word in words:
    print(word + " ----> " + stemming.stem(word))
# OUTPUT:
# eating -> eat
# eats -> eat
# eaten -> eaten
# writing -> write
# writes -> write
# programming -> program
# study -> studi

# Import Regexp Stemmer
from nltk.stem import RegexpStemmer

# Initialize RegexpStemmer with regex pattern
# 'ing$|s$|e$|able$' → removes these suffixes only if found at end of word
# min=4 → word must have minimum length of 4
reg_stemmer = RegexpStemmer('ing$|s$|e$|able$', min=4)

print(reg_stemmer.stem('eating'))
# OUTPUT: eat

# Import Snowball Stemmer (more advanced than Porter)
from nltk.stem import SnowballStemmer

# Initialize Snowball Stemmer for English language
snowballstemmer = SnowballStemmer('english')

# Apply Snowball stemming
for word in words:
    print(word + " -> " + snowballstemmer.stem(word))
# OUTPUT: more linguistically accurate stems

# Import WordNet Lemmatizer
from nltk.stem import WordNetLemmatizer

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Lemmatization with POS tag ('v' = verb)
print(lemmatizer.lemmatize("going", pos='v'))
# OUTPUT: go

# Sentence for POS tagging
sent = "Everything will be fine soon she. Inshallah one day"

# Tokenize sentence into words
words = nltk.word_tokenize(sent)

# Remove English stopwords
from nltk.corpus import stopwords
words = [word for word in words if word not in set(stopwords.words('english'))]

# Assign Part-of-Speech tags
pos_tag = nltk.pos_tag(words)

print(pos_tag)
# OUTPUT: [('Everything', 'NN'), ('fine', 'JJ'), ('soon', 'RB'), ('Inshallah', 'NN'), ('one', 'CD'), ('day', 'NN')]

# Named Entity Recognition (NER)
nltk.download('words')

# Tokenize and POS tag sentence
words = nltk.word_tokenize(sent)
tag_elements = nltk.pos_tag(words)

# Perform Named Entity Recognition and visualize
nltk.ne_chunk(tag_elements).draw()
# OUTPUT: Opens a tree diagram showing named entities

# Paragraph for text preprocessing
paragraph = "Everything will be fine soon she. Inshallah one day"

# Initialize Porter Stemmer
stemmer = PorterStemmer()

# Sentence tokenization
sent = nltk.sent_tokenize(paragraph)

# Apply stopword removal + stemming
for i in range(len(sent)):
    words = nltk.word_tokenize(sent[i])
    words = [stemmer.stem(word) for word in words if word not in set(stopwords.words('english'))]
    sent[i] = ' '.join(words)

print(sent)
# OUTPUT: ['Everyth fine soon Inshallah one day']

# Import regex library
import re

# Empty corpus list
corpus = []

# Paragraph with numbers and special characters
paragraph = "Everything will be fine 367634@ soom inshallah"

# Sentence tokenization
sentence = nltk.sent_tokenize(paragraph)

# Text cleaning loop
for i in range(len(sentence)):
    review = re.sub('[^a-zA-Z]', ' ', sentence[i])  # Remove non-alphabet characters
    review = review.lower()                         # Convert text to lowercase
    corpus.append(review)

print(corpus)
# OUTPUT: ['everything will be fine  soom inshallah']

# Bag of Words using CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer

corp2 = ["I love NLP", "I love ML"]

# Initialize CountVectorizer
cv = CountVectorizer()

# Fit and transform text into BoW vectors
x = cv.fit_transform(corp2)

print(cv.get_feature_names_out())
# OUTPUT: ['love' 'ml' 'nlp']

print(x.toarray())
# OUTPUT:
# [[1 0 1]
#  [1 1 0]]

# TF-IDF Vectorization
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize TF-IDF Vectorizer
# ngram_range can be used like (2,2) or (3,3) for n-grams
vectorizer = TfidfVectorizer()

# Fit and transform text into TF-IDF matrix
X = vectorizer.fit_transform(corp2)

print(X.toarray())
# OUTPUT: TF-IDF weighted numerical vectors
```
