import jax.numpy as jnp
from jax import grad, jit, vmap, random

# Create random input and output data
key = random.PRNGKey(0)
x = random.normal(key, (10, 3))
y = random.normal(key, (10,))

# Initialize random weights
w = random.normal(key, (3,))

# Define loss function
def loss(w, x, y):
    return jnp.sum((jnp.dot(x, w) - y) ** 2)

# Compute gradient of loss w.r.t. w
grad_loss = grad(loss)

print(grad_loss(w, x, y)) # => [-0.066 -0.173 -0.043]

# Compile a faster version of grad_loss
fast_grad_loss = jit(grad(loss))
print(fast_grad_loss(w, x, y)) # => [-0.066 -0.173 -0.043]

# Vectorize the loss function; now expecting a batch of inputs x and y
vectorized_loss = vmap(loss, in_axes=(None, 0, 0))
print(vectorized_loss(w, x, y)) # => [0.202 0.167 0.230 0.199 0.068 0.140 0.112 0.243 0.168 0.177]

# Compute the gradient of the vectorized loss function
# (note that we're passing in batches of x and y!)
vectorized_grad_loss = vmap(grad(loss), in_axes=(None, 0, 0))
print(vectorized_grad_loss(w, x, y)) # => [[-0.066 -0.173 -0.043]
                                     #     [-0.049 -0.173 -0.082]
                                     #     [-0.073 -0.171 -0.033]
                                     #     [-0.061 -0.170 -0.053]
                                     #     [-0.063 -0.172 -0.055]
                                     #     [-0.070 -0.171 -0.028]
                                     #     [-0.067 -0.172 -0.041]
                                     #     [-0.067 -0.174 -0.042]
                                     #     [-0.070 -0.172 -0.035]
                                     #     [-0.064 -0.173 -0.048]]

# Compute the gradient