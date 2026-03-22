# Model 2: Long Short-Term Memory Network (LSTM)

## Model Architecture
This model utilizes a Recurrent Neural Network (RNN) variant to capture the sequential dependencies and context of words over long review texts.
1. **Embedding Layer:** Maps the 10,000 vocabulary words into 100-dimensional continuous vectors.
2. **LSTM Layer:** 64 recurrent units. This layer processes the sequences sequentially, maintaining a "memory" state to understand the context of the sentence from beginning to end.
3. **Dense Layer:** 32 units (ReLU) to process the LSTM output.
4. **Output Layer:** 1 unit (Sigmoid) for binary sentiment classification.

## Techniques Applied
* **Sequence Modeling:** Replaced the shallow MLP from Assignment 1 with an LSTM to solve the vanishing gradient problem associated with standard RNNs.
* **Recurrent Dropout:** Applied a dropout rate of 0.2 directly inside the LSTM layer to prevent co-adaptation of recurrent units.
* **Early Stopping:** Configured with a patience of 3 epochs to optimize training time and prevent overfitting on the validation set.
* **Pre-padding for Sequence Optimization:** Applied `padding='pre'` instead of post-padding. Because Recurrent Neural Networks process data sequentially, placing the zeros at the beginning ensures the model processes the actual words immediately before making a prediction, rather than reading the words and then forgetting the context after processing hundreds of trailing zeros.