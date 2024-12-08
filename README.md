# README
## Overview
This project provides a pipeline for emotion classification using a fine-tuned BERT model. The model can classify text data into specific emotional categories (e.g., Anger, Joy, Sadness, etc.) by leveraging pre-trained language representations from BERT and fine-tuning them on a labeled dataset.

## Requirements
Before running the code, ensure that you have the following packages installed in your Python environment:

```bash
PyTorch
Transformers
Pandas
Scikit-learn
```
You can install these using pip:

```bash
pip install torch transformers pandas scikit-learn
```
*Note: If you have a GPU, make sure to install the GPU-optimized version of PyTorch to speed up training and inference.

## Data Preparation
You will need a training and a test dataset in CSV format. The CSV files should contain at least the following:

A column with the text data to classify.
A corresponding label column indicating the emotion.
For example, github-train.csv and github-test.csv might look like this:

```csv
text,Anger
"This a buggy code, who wrote the smelly code",1
"The code is refactored properly, thumbs up",0
```

Note: The --col argument in the script corresponds to the label column that will be used for classification. In the above example, --col=Anger means the script will use the "Anger" column as the target for classification. Make sure your CSV's label column name matches the value you pass to --col.

## **Fine-Tuned BERT Model**

You need a fine-tuned BERT model to run the classification. The `--model` argument should point to a directory or a file path containing the following components:

1. **`config.json`**: Contains the configuration of the model architecture.
2. **`pytorch_model.bin`**: Stores the fine-tuned model weights.
3. **`vocab.txt`** (optional): The vocabulary file if a custom tokenizer is used.

### Example Directory Structure

If your fine-tuned model is stored in a directory named `fine_tuned_bert/`, it might look like this:

## **Running the Classification Script**

The main entry point for running the classification is `1_bert-classification.py`. This script performs the following tasks:

1. **Loads the fine-tuned BERT model.**
2. **Reads the training and test datasets.**
3. **Trains the classifier**
4. **Evaluates the model** on the test dataset.

---

### **Arguments**

| Argument       | Description                                                  |
|----------------|--------------------------------------------------------------|
| `--epoch`      | Number of training epochs.                                   |
| `--model`      | Path to the fine-tuned BERT model (directory or file).        |
| `--traindata`  | Path to the training data CSV.                               |
| `--testdata`   | Path to the test data CSV.                                   |
| `--col`        | Name of the column representing the emotion in the dataset.  |

---

### **Example**

Run the script with the following command:

```bash
python 1_bert-classification.py \
    --epoch 10 \
    --model fine_tuned_bert \
    --traindata=github-train.csv \
    --testdata=github-test.csv \
    --col=Anger
```
This command will run the script for 10 epochs using the model located in fine_tuned_bert/, training on github-train.csv, testing on github-test.csv, and using the "Anger" column as the emotion label.


Results
After running the script, it will print out evaluation metrics (e.g.,F1 score) for the test set. 

