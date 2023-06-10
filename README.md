# Ranking Critical Neurons in ResNet20 using PyTorch Hooks

This project focuses on ranking critical neurons in the ResNet20 model using PyTorch hooks. The ResNet20 model used in this project is sourced from the [akamaster/pytorch_resnet_cifar10](https://github.com/akamaster/pytorch_resnet_cifar10) repository.

## Problem Statement

The objective of this project is to perturb the values of each neuron in the ResNet20 model and measure the impact on the model's performance. By perturbing the weights of the neurons and observing the resulting loss, we can determine the critical neurons that have a significant impact on the model's performance.

## Implementation

The ranking of critical neurons is performed using the following code snippet:

```python
# Perturb the values of each neuron and measure the impact on the performance
scores = []
#with torch.no_grad(): tambah tab after for
for name, param in net.named_parameters():
    if 'weight' in name:
        # Save the original weights
        original_weights = param.data.clone()

        # Perturb the weights
        param.data += 0.1 
        #param.add_(0.1)

        # Forward pass
        output = net(images)
        loss = criterion(output, labels)

        # Compute the score (impact on the performance)
        score = loss.item()

        # Append the score to the list
        scores.append((name, score))

        # Restore the original weights
        param.data = original_weights

# Sort the scores and print the ranking
scores = sorted(scores, key=lambda x: x[1], reverse=True)
for i, (name, score) in enumerate(scores):
    print(f'{i+1}. {name} - {score:.4f}')
```

In this code, we iterate over each parameter in the ResNet20 model and check if it is a weight parameter. We save the original weights, perturb the weights by adding 0.1, perform a forward pass, compute the loss, and store the score (impact on performance) in a list. Finally, we sort the scores in descending order and print the ranking of critical neurons.

## Usage

To replicate the ranking of critical neurons or extend the functionality of this project, follow these steps:

1. Clone the [akamaster/pytorch_resnet_cifar10](https://github.com/akamaster/pytorch_resnet_cifar10) repository to obtain the ResNet20 model implementation.

2. Modify the code snippet provided above according to your needs. You can adjust the perturbation value, apply different criteria for ranking, or incorporate additional functionality.

3. Use the modified code within your PyTorch project to rank the critical neurons in the ResNet20 model.

4. Optionally, you can visualize the ranking results or analyze the impact of critical neurons on the model's performance.

5. Or you can just run `neuron_ranking.py`

## References

- [akamaster/pytorch_resnet_cifar10](https://github.com/akamaster/pytorch_resnet_cifar10)

Please refer to the original repository for more details on the ResNet20 model implementation.
