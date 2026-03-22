# Deep Sentiment Analyzer: IMDB Review Classification
This project builds, trains, and evaluates multiple Deep Learning architectures to classify IMDB movie reviews as positive or negative. The goal is to move beyond basic lexicon-based sentiment analysis and implement advanced neural network techniques to capture contextual information from text.

## Project Structure
The repository is organized by model architecture:
* `/Model_1_CNN`: Implementation of a 1D Convolutional Neural Network for feature extraction.
* `/Model_2_LSTM`: Implementation of a Long Short-Term Memory network for sequential context.
* `/Model_3_GRU_Word2Vec`: Implementation of a Gated Recurrent Unit network utilizing pretrained Word2Vec embeddings.

## Key Techniques Applied
* **Text Preprocessing:** Automated cleaning (HTML removal, punctuation stripping, lowercasing) and sequence padding.
* **Optimization:** 
    * **Pre-padding:** Used `padding='pre'` to ensure RNNs process meaningful text immediately before classification.
    * **Regularization:** Employed Dropout (0.3 - 0.5) and Early Stopping to prevent overfitting.
    * **Transfer Learning:** Integrated `Gensim` Word2Vec pretrained embeddings to initialize the GRU model with semantic word relationships.
    * **Dynamic Learning:** Implemented `ReduceLROnPlateau` to adjust the learning rate based on validation loss.
## How to Run
1. Ensure the `IMDB_Dataset.csv` is present in your environment.
2. Run the provided Python notebook `classifier2.ipynb` (or individual model scripts).
3. Check the respective folders for `results.txt` and confusion matrix visualizations.
