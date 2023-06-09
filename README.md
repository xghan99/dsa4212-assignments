# dsa4212-assignments
This repository contains some of the code I wrote for the assignments in DSA4212 - Optimisation for Large-Scale Data-Driven Inference.

## Convolutional Neural Networks (CNN)
The goal of this assignment was to improve the accuracy of a CNN in classifying images to one of ten categories, subjected to a compute constraint of 1 GPU and 120 seconds of training time. We experimented with different network architectures, hyperparamters and data augmentation strategies. We managed to attain a test accuracy of 81.76%, an improvement over the baseline of around 68%. Within the folder are some of my code contributions to the project.

Libraries Used: TensorFlow, Flax, Jax, Optax

## Collaborative Filtering
The goal of this assignment is to build a recommender system that can correctly predict the ratings that a user gives on an anime. We experimented with mean user models, mean anime models and matrix factorization models. These models were also ensembled to attempt to lower the mean squared error of the predictions. Similarly, the folder contains some of my code contributions to the project.

Libraries Used: Pandas, Numpy, Jax, Jaxopt

## Word Embeddings
The goal of this assignment is to build word embeddings from scratch, without the use of libraries such as Gensim or FastText. We explored the use of Continuous Bag-of-Words and Skip-gram to train word embeddings from a text dataset consisting of highlights from CNN and DailyMail (https://www.kaggle.com/datasets/gowrishankarp/newspaper-text-summarization-cnn-dailymail).

Libraries Used: Pandas, Jax
