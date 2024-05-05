## Gradient Checkpointing

Neural networks primarily use memory in two ways:

1. Storing model weights
2. During training:
    - Forward pass to compute and store activations of all layers
    - Backward pass to compute gradients at each layer

This restricts us from training larger models and also limits the max batch size that can potentially fit into memory

We know that in a neural network:
- The activations of a specific layer can be solely computed using the activations of the previous layer.
- Updating the weights of a layer only depends on two things:
    - The activations of that layer.
    - The gradients computed in the next (right) layer.


Gradient checkpointing exploits these ideas to optimize backpropagation:
- Divide the network into segments before backpropagation
- In each segment:
    - Only store the activations of the first layer.
    - Discard the rest of the activations.
- When updating the weights of layers in a segment, recompute its activations using the first layer in that segment.


Recomputing the activations only when they are needed tremendously reduces the memory requirement. Essentially, we don’t need to store all the intermediate activations in memory. This allows us to train the network on larger batches of data.

Typically, gradient checkpointing can reduce memory usage by 50-60%, which is massive. Of course, this does come at a cost of slightly increased run-time. This can typically range between 15-25%.

It is because we compute some activations twice. So there's always a tradeoff between memory and run-time.

Yet, gradient checkpointing is an extremely powerful technique to train larger models without resorting to more intensive techniques like distributed training, for instance.

The code to implement is under study.
Another add on comment