# Model 1: 1D Convolutional Neural Network (1D CNN)

## Model Architecture
This model replaces the Lexicon/MLP approach from Assignment 1 with a Deep Learning approach designed for sequence processing. The architecture consists of:
1. **Embedding Layer:** Learns dense vector representations (size 100) for the top 10,000 words in the vocabulary.
2. **Conv1D Layer:** Uses 128 filters with a kernel size of 5 to detect local n-gram patterns (e.g., phrases like "not good" or "absolutely loved").
3. **GlobalMaxPooling1D Layer:** Extracts the most salient features from the feature maps.
4. **Dense Layer:** 64 units with ReLU activation for intermediate processing.
5. **Output Layer:** A single unit with a Sigmoid activation function to output a binary sentiment probability.

## Techniques Applied
* **Text Tokenization & Padding:** Transformed raw text into standardized sequences of length 200.
* **Dropout (0.5):** Applied before the final layer to randomly zero out neurons, preventing the model from overfitting to the training data.
* **Early Stopping:** Monitored the validation loss during training and automatically stopped training when the model stopped improving for 3 consecutive epochs, restoring the best weights.