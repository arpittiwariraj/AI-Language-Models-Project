# AI Language Models - Transformer-Based NLP

## Project Description

This project implements custom AI language models using the **transformer architecture** for **text classification** and **text generation** tasks. Transformers have revolutionized natural language processing (NLP) by enabling models to understand and generate human-like text. In this project, a simplified transformer model is built from scratch using **TensorFlow** to handle various NLP tasks such as generating text and classifying sentences.

## Features

- **Custom Transformer Model**: A transformer-based architecture is implemented from scratch using TensorFlow.
- **Text Classification and Generation**: The model can be trained on any text dataset to classify sentences or generate new text based on input prompts.
- **Flexible Data Handling**: Data preprocessing utilities allow for easy loading and cleaning of text datasets.
- **Modular Code Structure**: The project is organized into well-structured scripts, making it easy to extend or modify components like the model architecture or preprocessing functions.

## Installation

1. **Clone this repository**:
   ```bash
   git clone https://github.com/arpittiwariraj/AI-Language-Models-Project.git
   cd AI-Language-Models-Project
2. **Install the required dependencies**
   pip install -r requirements.txt
3. **To train the Model**
   ```bash
   python scripts/train.py
5. **To Generate Text**
   ```bash
   python scripts/generate.py
7. **Project Structure**
   ```bash
   AI-Language-Models-Project/
```bash│
├── data/                    # Dataset files (train and validation)
├── notebooks/               # Jupyter notebooks for experiments
├── scripts/                 # Python scripts for training and generating text
│   ├── train.py             # Script for training the transformer model
│   ├── generate.py          # Script for generating text using the model
│   ├── transformer_model.py # Defines the transformer model architecture
├── utils/                   # Helper functions for data preprocessing
│   └── data_preprocessing.py # Functions for loading and preprocessing text data
├── models/                  # Saved model files
├── README.md                # Description and instructions for the project
├── requirements.txt         # List of required Python libraries
└── .gitignore               # Files to be ignored by Git
