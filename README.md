# [Centimani](https://www.usenix.org/conference/atc24/presentation/xie): Enabling Fast AI Accelerator Selection for DNN Training with a Novel  Performance Predictor

The micro-benchmarks is to benchmark operations that are important to deep learning on different hardware platforms.

# Types of Operations

## Convolution Layers

Convolutions make up the vast majority of flops in networks that operate on images and videos and form important parts of networks such as speech and natural language modeling, thus making them perhaps the single most important layer from a performance perspective.

## ReLU Layers

The Rectified Linear Unit (ReLU) is the most commonly used activation function in deep learning models.

## Recurrent Layers

Recurrent neural networks (RNN) are a class of neural networks that is powerful for modeling sequence data such as time series or natural language. There are three types of recurrent cells, such as, vanilla RNNs, LSTMs and GRUs. We use the LSTM as a representative.

## Transformer

Transformer is a component used in many neural network designs for processing sequential data, such as natural language text, genome sequences, sound signals or time series data.

## All-Reduce

Neural networks today are often trained across multiple GPUs or even multiple systems, each with multiple GPUs. There are two main categories of techniques for doing this: synchronous and asynchronous. Synchronous techniques rely on keeping the parameters on all instances of the model synchronized, usually by making sure all instances of the model have the same copy of the gradients before taking an optimization step. The primitive usually used to perform this operation is called All-Reduce.

All-Reduce based on the number of ranks and the size of the data.
