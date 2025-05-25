# README

## Overview
This project provides a pipeline for emotion and incivility classification using a fine-tuned BERT and CodeBert model. The model can classify text data into specific emotional categories (e.g., Anger, Joy, Sadness, etc.) and incivility categories (Civil, Uncivil) by leveraging pre-trained language representations from BERT and CodeBERT, and fine-tuning them on labeled datasets.

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

## Dataset
1. Original Dataset for Fine-tuning (Given in the paper): `Fig_Lan_Annotation.csv` (contains 1662 Figurative sentence example)
2. Augmented Dataset for Fine-tuning (Using ChatGPT): `data.csv` (contains 9976 Figurative sentence example)
3. Github Emotion dataset: `github-train.csv` for training and `github-test.csv` for validation.
4. Incivility Dataset: `incivility-train-data.csv` for training and `incivility-test-data.csv` for validation


## Fine Tuning Model
### Fine Tuning Bert Model
- Use the following command to fine tune Bert Model with the dataset `Fig_Lan_Annotation.csv`.
```bash
python 1_fine_tune_bert.py \
--epoch 2 \
--data Fig_Lan_Annotation.csv \
--output 2_test
```

*Note: `--epoch` argument for the number of epochs, `--data` for path of the dataset and `--output` for the path where the fine-tuned model will be stored

- Use the following command to fine tune Bert Model with the dataset `data.csv`.
```bash
python 1_fine_tune_bert.py \
--epoch 2 \
--data data.csv \
--output 2_test
```
*Note: `--epoch` argument for the number of epochs, `--data` for path of the dataset and `--output` for the path where the fine-tuned model will be stored

### Fine Tuning CodeBert Model
- Use the following command to fine tune CodeBert Model with the dataset `Fig_Lan_Annotation.csv`.
```bash
python 2_fine_tune_codebert.py \
--epoch 2 \
--data Fig_Lan_Annotation.csv \
--output 2_test
```

*Note: `--epoch` arguments for the number of epochs, `--data` for path of the dataset and `--output` for the path where the fine-tuned model will be stored

- Use the following command to fine tune CodeBert Model with the dataset `data.csv`.
```bash
python 2_fine_tune_codebert.py \
--epoch 2 \
--data data.csv \
--output 2_test
```
*Note: `--epoch` arguments for the number of epochs, `--data` for path of the dataset and `--output` for the path where the fine-tuned model will be stored


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

## Github Emotion Classification
### 1. Running the Fine-Tune Bert Model
The main entry point for running the classification is `1_bert-classification.py`. This script performs the following tasks:

1. **Loads the fine-tuned BERT model.**
2. **Reads the training and test datasets.**
3. **Trains the classifier**
4. **Evaluates the model** on the test dataset.

--

##### **Arguments**

| Argument       | Description                                                  |
|----------------|--------------------------------------------------------------|
| `--epoch`      | Number of training epochs.                                   |
| `--model`      | Path to the fine-tuned BERT model (directory or file).        |
| `--traindata`  | Path to the training data CSV.                               |
| `--testdata`   | Path to the test data CSV.                                   |
| `--col`        | Name of the column representing the emotion in the dataset.  |

---

Since fine-tuning takes more time, you can directly use my fine-tuned models. Run the following script to get the results.

- Get Classification results for the Bert model fine-tuned with the dataset `Fig_Lan_Annotation.csv`
```bash
python 1_bert-classification.py \
    --epoch 100 \
    --model fine_tuned_bert_less_data \
    --traindata=github-train.csv \
    --testdata=github-test.csv \
    --col=Anger
```

- Get Classification results for the Bert model fine-tuned with the dataset `data.csv`
```bash
python 1_bert-classification.py \
    --epoch 100 \
    --model fine_tuned_bert \
    --traindata=github-train.csv \
    --testdata=github-test.csv \
    --col=Anger
```

Note: The following scripts will give you the results for the emotion `Anger`. Please change the value if you want to get the results for other emotions (Joy, Fear, Sadness, Love, Surprise)


### 2. Running the Fine-Tune CodeBert Model
The main entry point for running the classification is `2_codebert-classification.py`. This script performs the following tasks:

1. **Loads the fine-tuned CodeBERT model.**
2. **Reads the training and test datasets.**
3. **Trains the classifier**
4. **Evaluates the model** on the test dataset.

--

##### **Arguments**

| Argument       | Description                                                  |
|----------------|--------------------------------------------------------------|
| `--epoch`      | Number of training epochs.                                   |
| `--model`      | Path to the fine-tuned Code BERT model (directory or file).        |
| `--traindata`  | Path to the training data CSV.                               |
| `--testdata`   | Path to the test data CSV.                                   |
| `--col`        | Name of the column representing the emotion in the dataset.  |

---

Since fine-tuning takes more time, you can directly use my fine-tuned models. Run the following script to get the results.

- Get Classification results for the CodeBert model fine-tuned with the dataset `Fig_Lan_Annotation.csv`
```bash
python 2_codebert-classification.py \
    --epoch 100 \
    --model fine_tuned_code_bert_less_data \
    --traindata=github-train.csv \
    --testdata=github-test.csv \
    --col=Anger
```

- Get Classification results for the CodeBert model fine-tuned with the dataset `data.csv`
```bash
python 2_codebert-classification.py \
    --epoch 100 \
    --model 3_fine_tuned_code_bert \
    --traindata=github-train.csv \
    --testdata=github-test.csv \
    --col=Anger
```

*Note: The following scripts will give you the results for the emotion `Anger`. Please change the value if you want to get the results for other emotions (Joy, Fear, Sadness, Love, Surprise)

## Github Incivility Classification
### 1. Running the Fine-Tune Bert Model
The main entry point for running the classification is `1_bert-classification.py`. This script performs the following tasks:

1. **Loads the fine-tuned BERT model.**
2. **Reads the training and test datasets.**
3. **Trains the classifier**
4. **Evaluates the model** on the test dataset.

--

##### **Arguments**

| Argument       | Description                                                  |
|----------------|--------------------------------------------------------------|
| `--epoch`      | Number of training epochs.                                   |
| `--model`      | Path to the fine-tuned BERT model (directory or file).        |
| `--traindata`  | Path to the training data CSV.                               |
| `--testdata`   | Path to the test data CSV.                                   |
| `--col`        | Name of the column representing the emotion in the dataset.  |

---

Since fine-tuning takes more time, you can directly use my fine-tuned models. Run the following script to get the results.

- Get Classification results for the Bert model fine-tuned with the dataset `Fig_Lan_Annotation.csv`
```bash
python 1_bert-classification.py \
    --epoch 100 \
    --model fine_tuned_bert_less_data \
    --traindata=incivility-train-data.csv \
    --testdata=incivility-test-data.csv \
    --col=label
```

- Get Classification results for the Bert model fine-tuned with the dataset `data.csv`
```bash
python 1_bert-classification.py \
    --epoch 100 \
    --model fine_tuned_bert \
    --traindata=incivility-train-data.csv \
    --testdata=incivility-test-data.csv \
    --col=label
```

### 2. Running the Fine-Tune CodeBert Model
The main entry point for running the classification is `2_codebert-classification.py`. This script performs the following tasks:

1. **Loads the fine-tuned CodeBERT model.**
2. **Reads the training and test datasets.**
3. **Trains the classifier**
4. **Evaluates the model** on the test dataset.

--

##### **Arguments**

| Argument       | Description                                                  |
|----------------|--------------------------------------------------------------|
| `--epoch`      | Number of training epochs.                                   |
| `--model`      | Path to the fine-tuned Code BERT model (directory or file).        |
| `--traindata`  | Path to the training data CSV.                               |
| `--testdata`   | Path to the test data CSV.                                   |
| `--col`        | Name of the column representing the emotion in the dataset.  |

---

Since fine-tuning takes more time, you can directly use my fine-tuned models. Run the following script to get the results.

- Get Classification results for the CodeBert model fine-tuned with the dataset `Fig_Lan_Annotation.csv`
```bash
python 2_codebert-classification.py \
    --epoch 100 \
    --model fine_tuned_code_bert_less_data \
    --traindata=incivility-train-data.csv \
    --testdata=incivility-test-data.csv \
    --col=label
```

- Get Classification results for the CodeBert model fine-tuned with the dataset `data.csv`
```bash
python 2_codebert-classification.py \
    --epoch 100 \
    --model 3_fine_tuned_code_bert \
    --traindata=incivility-train-data.csv \
    --testdata=incivility-test-data.csv \
    --col=label
```

