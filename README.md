---

# Screenplay Generator

This project is a simple transformer-based language model built with PyTorch to generate screenplay text. The model is trained on text data to predict the next token in a sequence, allowing it to generate coherent text in a screenplay format. This project uses Streamlit for a simple, interactive interface where you can train the model, view the training progress, and generate new text.

## Features

- **Training & Evaluation**: Train the model on provided text data with options to view loss over iterations.
- **Text Generation**: Generate new screenplay text based on a trained model.
- **Interactive Interface**: Easily train and generate text from a user-friendly Streamlit interface.

## Requirements

- Python 3.8 or later
- PyTorch
- Streamlit

### Install Dependencies

To install the required libraries, run:

```bash
pip install torch streamlit
```

## How to Use

1. **Prepare the Training Data**: Ensure you have a text file (`tiny1.txt`) that contains your training data in the same directory. This file will be used as the input for the model.

2. **Run the Streamlit App**: Launch the app by running the following command in the terminal:

   ```bash
   streamlit run screenplayGPT.py
   ```

3. **Use the Interface**:

   - **View Text**: Expand the "View Text" section to read the training data.
   - **Create Model**: Click on "Bigram Model" to display the number of parameters and the optimizer being used.
   - **Train Model**: Press "Train" to start training the model. The training loss and validation loss will be displayed every 100 iterations.
   - **Generate Text**: Once trained, click on "Generate" to produce a block of generated text. The model will generate a sequence of tokens based on the training data.

## Model Architecture

The model is a multi-head self-attention Transformer composed of multiple layers, each with a feed-forward network and layer normalization. The architecture consists of:
   
- **Multi-Head Attention**: The model uses 4 attention heads with a 64-dimensional embedding.
- **Transformer Blocks**: 4 layers of transformer blocks, each including a feed-forward layer and a self-attention mechanism.
- **Dropout**: Applied throughout the model to prevent overfitting during training.
- **Token Embedding and Positional Embedding**: Each token and position in the sequence is embedded into a 64-dimensional vector space.

## Hyperparameters

The model hyperparameters are configured as follows:
- `batch_size`: 16
- `block_size`: 32
- `max_iters`: 2000
- `learning_rate`: 1e-3
- `n_embd`: 64
- `n_head`: 4
- `n_layer`: 4

These values can be adjusted within the script for experimentation.

## Code Structure

- **Data Loading**: The text data is tokenized, split into training and validation sets, and loaded in batches for training.
- **Model Training**: A simple loop to train the model using backpropagation and an AdamW optimizer. Loss is evaluated on training and validation sets periodically.
- **Text Generation**: The `generate` function allows the model to predict the next token, extending the sequence until a set limit (`max_new_tokens`) is reached.

## Future Improvements

- **Add More Data**: The current setup uses a small dataset for demonstration. Adding more data can help improve the model's output quality.
- **Experiment with Hyperparameters**: Try different values for `n_embd`, `n_head`, and `n_layer` to achieve better performance.
- **Model Saving and Loading**: Add functionality to save the trained model for reuse and load pre-trained models.

## Acknowledgments

Inspired by Andrej Karpathy’s [char-rnn](https://github.com/karpathy/char-rnn) project. The project implements a transformer-based approach instead of the traditional RNN.

---
