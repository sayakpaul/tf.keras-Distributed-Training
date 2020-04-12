# tf.keras-Distributed-Training

Accompanies with [this report](https://app.wandb.ai/sayakpaul/tensorflow-multi-gpu-dist/reports/Distributed-training-in-tf.keras-with-W%26B--Vmlldzo3NzUyNA). 

![](https://i.ibb.co/t8PyVQW/Screen-Shot-2020-04-12-at-10-12-29-AM.png)

This repository shows how to seamlessly integrate [`tf.distribute.MirroredStrategy`](https://www.tensorflow.org/api_docs/python/tf/distribute/MirroredStrategy) for distributing your training workloads across multiple GPUs for `tf.keras` models. Distributed training can be particularly very useful when you have very large datasets and the need to scale the training costs becomes very prominent with that. It becomes unrealistic to perform the training on only a single hardware accelerator (a GPU in this case), hence the need for performing distributed training. 

TensorFlow's [distributed strategies](https://www.tensorflow.org/api_docs/python/tf/distribute) make it extremely easier for us to seamlessly scale up our heavy training workloads across multiple hardware accelerators -- be it GPUs or even TPUs. That said, distributed training has been a challenge for a long time especially when it comes to neural network training. The primary challenges that come with distributed training procedures are as follows:
- How are we going to distribute the model parameters across the different devices? 
- How are we going to accumulate the gradients during backpropagation? 
- How are the model parameters going to be updated? 
  
All of these may sound very daunting if you think of the training process end-to-end. Thankfully, libraries like TensorFlow give us the freedom of incorporating distributed training very easily  -- be it for `tf.keras` models with the classic `fit` and `compile` paradigm or be it for custom training loops.  This report, however, only deals with the former. If you are interested in learning more about distributed training for custom training loops, be sure to check [this tutorial](https://www.tensorflow.org/tutorials/distribute/custom_training) out. 

## Dataset used

Cats vs. Dogs

## Hardware

- n1-standard-4vCPUs-15 GB
- 4 Tesla k80s
- Preconfigured Image: TensorFlow 2.1 (with Intel MKL-DNN/MKL and CUDA 10.1)

## Acknowledgements

- ML-GDE Program (know about the GDE program [here](https://developers.google.com/community/experts)) for allowing me GCP Cloud Credits otherwise, these experiments (all of them are done on GCP) wouldn't have been possible. 
- [Martin Gorner](https://twitter.com/martin_gorner) for his guidance. 
