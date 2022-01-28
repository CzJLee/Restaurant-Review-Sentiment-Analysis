# Restaurant Review Sentiment Analysis

Predicting the star rating of a text review using Natural Language Processing (NLP) Deep Learning models. This project uses Text Vectorization, LSTM layers, Transformers, and the TF Dataset API to analyse the sentiment and predict the star rating of Yelp restaurant reviews. 

- [Restaurant Review Sentiment Analysis](#restaurant-review-sentiment-analysis)
	- [Dataset](#dataset)
	- [Project Goal](#project-goal)
	- [Project Contents](#project-contents)
		- [dataset.ipynb](#datasetipynb)
		- [train_vectorized.ipynb](#train_vectorizedipynb)
		- [train_sequence.ipynb](#train_sequenceipynb)
		- [train_transformer.ipynb](#train_transformeripynb)
		- [fast_train_vectorized.ipynb](#fast_train_vectorizedipynb)
		- [train_large_model.ipynb](#train_large_modelipynb)
	- [Project Accomplishments](#project-accomplishments)
	- [Results](#results)

## Dataset

This project uses the Yelp Open Dataset. This dataset contains information on businesses, users, reviews, and more. Only the text and star ratings from the review dataset is used. 

The Yelp Open Dataset can be downloaded directly from Yelp.

https://www.yelp.com/dataset

## Project Goal
Using NLP and Sentiment Analysis, predict the star rating using various ML models including 
- Bag of Words Models
- Sequence Models
- Transformer Models

## Project Contents

### [dataset.ipynb](dataset.ipynb)

- This notebook creates a subset of the large review dataset to use for training. 

### [train_vectorized.ipynb](train_vectorized.ipynb)

- This is the first step and simplest model using Text Vectorization and NGrams to classify text. 

### [train_sequence.ipynb](train_sequence.ipynb)

- The next step is to build a model that processes text as a sequence. This is done an Embedding layer and Bidirectional LSTM layers. 

### [train_transformer.ipynb](train_transformer.ipynb)

- Transformers are a new method of encoding text that is able to combine the benefits of older methods. This notebook trains a model with custom Transformer encoding layers. 

### [fast_train_vectorized.ipynb](fast_train_vectorized.ipynb)

- I wanted to experiment with training on Google Colab cloud TPUs, so I dug deep into the TF Dataset API to try and improve the data throughput. Implementing these methods increased the epoch speed by about 20%. 

### [train_large_model.ipynb](train_large_model.ipynb)

- The final step is to try and train on the entire review dataset. This dataset is too large to fit into memory and process. To resolve this, and to use TF Record files, I built a custom TF Sharded Record Writer class. I then adjusted the training process to handle a model with a large amount of parameters and a large dataset. 

## Project Accomplishments

There were multiple tasks that I accomplished in this project
- Training on TPUs using Google Colab cloud TPU environment
- Read JSON in chunks using Pandas to create a subset of a larger dataset too large to load into memory
- Implement and test different Text Vectorizations for Natural Language Processing (NLP) including Bag of Words, Bigram Vectorization, and Term Frequency Inverse Document Frequency (TF-IDF) 
- Construct a sequential text model using an Embedding layer and Bidirectional LSTM layers
- Use Transformer layers for text encoding
- Use TF Dataset API to optimize data pipeline, and speedup TPU training
- Build a custom TF Sharded Record Writer class to shard a large dataset into multiple TF Record files
- Use TF Datasets to create balanced Train, Validation, and Test datasets
- Develop a training pipeline to train a large dataset on Google Colab cloud GPUs

## Results

The performance metrics of various models can be seen in the [model metrics text file](model_metrics.txt).

The best performance was achieved by the Embedded Sequence model, with a Mean Average Error of 0.309 and Mean Squared Error of 0.376. 