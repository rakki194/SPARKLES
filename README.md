# SPARKLES Optimizer

A powerful and adaptive optimization algorithm that combines several advanced techniques for improved neural network training.

## Overview

SPARKLES (**S**tochastic **P**arameter **A**djustment with **R**andomized **K**ick for **L**earning **E**nhancement **S**trategy) is an optimization algorithm that integrates multiple state-of-the-art optimization techniques, including gradient centralization, adaptive normalization, momentum amplification, adaptive step sizing, and stochastic updates. It's designed to provide faster convergence, improved stability, and better generalization in deep learning tasks.

**IMPORTANT: SPARKLES is specifically designed to work with bfloat16 (BF16) precision. Using full BF16 format for both model weights and input data is essential for optimal performance. Without BF16 precision, the optimizer's performance will be significantly degraded.**

BF16's numerical properties create the ideal balance for SPARKLES' stochastic operations. The reduced mantissa precision creates a beneficial noise floor that works synergistically with the randomized gradient updates, helping escape local minima while maintaining training stability. The optimizer's normalization and centralization techniques were specifically calibrated for BF16's characteristics and don't function correctly with other precision formats.

## Quick Start

```python
import torch
from sparkles import SPARKLES

# Define your model
model = YourModel()

# Create optimizer
optimizer = SPARKLES(
    model.parameters(),
    lr=1e-3,
    betas=(0.9, 0.999),
    centralization=0.5,
    normalization=0.5,
    amp_fac=2.0,
    stochastic_threshold=1e-6,
    permutation_strategy="magnitude"  # Use magnitude-aware permutation
)

# Training loop (recommended to use BF16 for optimal performance)
model = model.to(torch.bfloat16)
for epoch in range(num_epochs):
    for data, target in dataloader:
        data = data.to(torch.bfloat16)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_function(output, target)
        loss.backward()
        optimizer.step()
```

## Key Features

### 1. Gradient Centralization

Removes the mean of gradients for each layer to reduce internal covariate shift:

```plaintext
g_t = g_t - α_c · mean(g_t)
```

Benefits:

- Reduces internal covariate shift
- Improves training stability
- Enhances generalization

### 2. Adaptive Gradient Normalization

Normalizes gradients using their standard deviation with interpolation control:

```plaintext
g_t = (1 - α_n) · g_t + α_n · (g_t / std(g_t))
```

Options:

- Channel-wise or global normalization
- Controllable interpolation factor

### 3. Momentum with Amplification

Accelerates training in relevant directions:

```plaintext
m_t = β₁ · m_{t-1} + (1 - β₁) · g_t
g_t = g_t + γ · m_t
```

Features:

- Maintains exponential moving average of gradients
- Amplifies current gradient using momentum history
- Controlled by amplification factor (γ)

### 4. Adaptive Step Sizes

Adapts learning rates per parameter using moment estimation:

```plaintext
v_t = β₂ · v_{t-1} + (1 - β₂) · g_t²
m̂_t = m_t / (1 - β₁ᵗ)
v̂_t = v_t / (1 - β₂ᵗ)
η_t = lr / (1 - β₁ᵗ)
```

### 5. Stochastic Update Strategy

Helps escape local minima and saddle points with multiple permutation strategies:

```plaintext
if ||g_t - g_{t-1}|| < threshold:
    update = R(update)  # Apply stochastic operator R with selected strategy
```

Available permutation strategies:

- **Global** (default): Randomly permutes all elements, providing maximum exploration
- **Magnitude-aware**: Only permutes elements within similar magnitude bands, preserving scale relationships
- **Local neighborhood**: Permutes elements only within local regions, preserving structural relationships
- **Adaptive**: Varies permutation strength based on gradient variance, applying more randomization when updates are uniform

Features:

- Detects when gradients become small between iterations
- Applies a randomized operation to updates to escape potential local minima
- Controlled by stochastic threshold parameter
- Intelligently controls permutation aggressiveness based on the selected strategy

### 6. Stochastic BF16 Rounding

Enhances BF16 precision conversion with controlled randomness (enabled by default):

```plaintext
# Bit manipulation approach (default)
random_bits = random_int(0, 2^16 - 1)
result = ((x_fp32 + random_bits) & 0xFFFF0000) as BF16

# Alternative approach
x_bf16 = x.to(bf16)
if random() < probability:
    x_bf16 += noise  # Controlled noise based on BF16 epsilon
```

Features:

- Uses bit-level manipulation to achieve true stochastic rounding
- Adds carefully calibrated randomness during precision conversion
- Works synergistically with BF16's reduced mantissa precision
- Helps escape local minima through micro-perturbations
- Enabled by default for optimal performance
- Reduces memory usage by 50% compared to FP32 training
- Prevents stale gradients during long training runs

## Acknowledgments

### Compass Optimizer

SPARKLES draws significant inspiration from the [Compass optimizer](https://github.com/lodestone-rock/compass_optimizer) created by [lodestone-rock](https://github.com/lodestone-rock). The Compass optimizer introduced several innovative concepts that SPARKLES builds upon:

1. **Gradient Centralization Technique**: Compass pioneered the implementation of gradient centralization in an adaptive optimizer, which helps remove the mean of gradients to improve training stability.

2. **Momentum Amplification**: The concept of amplifying gradients with momentum history originated in Compass with its elegant implementation:

   ```python
   ema.mul_(beta1).add_(grad, alpha=1 - beta1)
   grad.add_(ema, alpha=amplification_factor)
   ```

3. **Simplified Bias Correction**: Compass implemented an efficient bias correction technique for adaptive step sizes that SPARKLES adopts and extends.

4. **Decoupled Weight Decay**: The approach to separating weight decay from gradient updates in Compass inspired SPARKLES' implementation of weight decay.

5. **Stochastic BF16 Rounding**: The bit manipulation approach for true stochastic rounding is adapted from lodestone-rock's implementation.

The clean, efficient implementation of the Compass optimizer provided an excellent foundation that SPARKLES extends with additional features like stochastic updates and gradient normalization. We're grateful to Lode for their contributions to optimization research.

### Torchastic

SPARKLES also builds upon concepts from [Torchastic](https://github.com/lodestone-rock/torchastic/), another innovative project by lodestone-rock that focuses on stochastic BF16-based optimization. Key inspirations include:

1. **Memory Efficiency**: Torchastic demonstrated that using BF16 for all operations (parameters, gradients, and optimizer states) can reduce memory consumption by 50% compared to FP32 training, enabling larger models or increased batch sizes within the same memory budget.

2. **Stochastic Rounding to Prevent Stale Gradients**: The stochastic rounding mechanism in SPARKLES is inspired by Torchastic's approach to solving the "stale gradient" problem, where small updates might be lost due to BF16's reduced mantissa precision. Stochastic rounding ensures that even tiny updates have a chance to affect the model parameters.

3. **Bit-Level Manipulation**: The implementation of adding randomness at the bit level (rather than just adding noise after conversion) was inspired by Torchastic's elegant solution.

We extend our thanks for demonstrating how stochastic BF16 training can be both memory-efficient and effective for preventing numerical instabilities during long training runs.

## Related Work

SPARKLES builds on the shoulders of several pioneering optimization algorithms in deep learning:

### SADAM

SPARKLES incorporates stochastic update strategies inspired by SADAM [7], which introduces randomness into the optimization process to help escape local minima and saddle points. This approach is particularly effective in complex loss landscapes.

### ADOPT

SPARKLES incorporates key insights from ADOPT (Adaptive Optimization with Provable convergence guarantees), which provides theoretical convergence guarantees regardless of β₂ value [1]. ADOPT introduced gradient clipping based on step count to improve numerical stability, which SPARKLES adopts through its `clip_lambda` parameter.

### Adam and AdamW

The adaptive moment estimation approach in SPARKLES is inspired by the Adam optimizer [2], which combines momentum and RMSProp techniques. Like AdamW [3], SPARKLES implements decoupled weight decay to better handle regularization by separating weight decay from gradient-based updates.

### Gradient Centralization

The gradient centralization component builds upon the work of Yong et al. [4], who demonstrated that removing the mean from gradients can significantly improve training stability and model generalization by reducing internal covariate shift.

### Momentum and RMSProp

The momentum amplification technique extends traditional momentum approaches [5] by adding an amplification factor to accelerate convergence. The adaptive step sizing draws from RMSProp [6], using running averages of squared gradients to normalize updates on a per-parameter basis.

### References

[1] Taniguchi, S., Harada, K., Minegishi, G., Oshima, Y., Jeong, S.C., Nagahara, G., Iiyama, T., Suzuki, M., Iwasawa, Y., and Matsuo, Y. (2024). "ADOPT: Modified Adam Can Converge with Any β2 with the Optimal Rate." In Advances in Neural Information Processing Systems.

[2] Kingma, D. P., & Ba, J. (2014). "Adam: A Method for Stochastic Optimization." arXiv preprint arXiv:1412.6980.

[3] Loshchilov, I., & Hutter, F. (2019). "Decoupled Weight Decay Regularization." In International Conference on Learning Representations.

[4] Yong, H., Huang, J., Hua, X., & Zhang, L. (2020). "Gradient Centralization: A New Optimization Technique for Deep Neural Networks." In European Conference on Computer Vision.

[5] Sutskever, I., Martens, J., Dahl, G., & Hinton, G. (2013). "On the importance of initialization and momentum in deep learning." In International Conference on Machine Learning.

[6] Tieleman, T., & Hinton, G. (2012). "Lecture 6.5-rmsprop: Divide the gradient by a running average of its recent magnitude." COURSERA: Neural Networks for Machine Learning.

[7] Chen, L., Tian, Y., Shi, Z., Zhang, D., & Zhu, P. (2022). "Stochastic Adam Method." arXiv preprint arXiv:2205.10247.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lr` | 1e-3 | Learning rate |
| `betas` | (0.9, 0.999) | Coefficients for computing running averages of gradient and its square |
| `eps` | 1e-8 | Term added to denominator for numerical stability |
| `weight_decay` | 0.0 | Weight decay (L2 penalty) |
| `centralization` | 0.5 | Strength of gradient centralization |
| `normalization` | 0.5 | Interpolation factor for normalized gradients |
| `normalize_channels` | True | Whether to normalize gradients channel-wise |
| `amp_fac` | 2.0 | Amplification factor for the momentum term |
| `clip_lambda` | λ(step) = step^0.25 | Function computing gradient clipping threshold |
| `decouple_weight_decay` | False | Whether to apply weight decay directly to weights |
| `clip_gradients` | False | Whether to enable gradient clipping |
| `stochastic_threshold` | 1e-6 | Threshold for applying stochastic updates |
| `use_stochastic_rounding` | True | Whether to apply stochastic rounding when converting to BF16 |
| `stochastic_rounding_prob` | 0.5 | Probability of applying noise in stochastic rounding |
| `stochastic_rounding_magnitude` | None | Magnitude of noise to apply (None = use BF16 epsilon) |
| `use_bit_manipulation` | True | Whether to use bit manipulation for stochastic rounding |
| `permutation_strategy` | "global" | Permutation strategy ("global", "magnitude", "local", "adaptive") |
| `magnitude_bands` | 5 | Number of magnitude bands for magnitude-aware shuffling |
| `local_neighborhood_size` | 0.1 | Size of local neighborhood as fraction of tensor size |
| `adaptive_scale_factor` | 1.0 | Scaling factor for adaptive permutation based on gradient variance |

## Advanced Usage

### Custom Gradient Clipping

```python
# Create custom clipping function based on step count
def custom_clip(step):
    return min(10.0, 0.1 * step**0.5)

optimizer = SPARKLES(
    model.parameters(),
    lr=1e-3,
    clip_lambda=custom_clip,
    clip_gradients=True
)
```

### Decoupled Weight Decay

```python
# Use decoupled weight decay for better regularization
optimizer = SPARKLES(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-4,
    decouple_weight_decay=True
)
```

### BF16 Training (Recommended)

```python
# Use BF16 format for optimal performance
model = model.to(torch.bfloat16)
optimizer = SPARKLES(model.parameters(), lr=1e-3)

for epoch in range(num_epochs):
    for data, target in dataloader:
        data = data.to(torch.bfloat16)
        # Rest of training loop
```

### Stochastic BF16 Rounding Options

```python
# Disable stochastic rounding (not recommended)
optimizer = SPARKLES(
    model.parameters(),
    lr=1e-3,
    use_stochastic_rounding=False
)

# Use noise-based approach instead of bit manipulation
optimizer = SPARKLES(
    model.parameters(),
    lr=1e-3,
    use_bit_manipulation=False,
    stochastic_rounding_prob=0.3,  # Lower probability of applying noise
    stochastic_rounding_magnitude=1e-4  # Custom noise magnitude
)
```

### Advanced Permutation Strategies

```python
# Magnitude-aware shuffling (only permutes elements with similar magnitudes)
optimizer = SPARKLES(
    model.parameters(),
    lr=1e-3,
    permutation_strategy="magnitude",
    magnitude_bands=10  # Use more bands for finer granularity
)

# Local neighborhood shuffling (maintains structural relationships)
optimizer = SPARKLES(
    model.parameters(),
    lr=1e-3,
    permutation_strategy="local",
    local_neighborhood_size=0.05  # Smaller local regions
)

# Adaptive permutation (varies strength based on gradient variance)
optimizer = SPARKLES(
    model.parameters(),
    lr=1e-3,
    permutation_strategy="adaptive",
    adaptive_scale_factor=2.0  # More aggressive scaling
)
```

## License

MIT
