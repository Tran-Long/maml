from functools import partial
import jax
from flax.training import train_state
from typing import Any
import optax
import jax.numpy as jnp
import flax.linen as nn

class TrainStateWithBatchNorm(train_state.TrainState):
    batch_stats: Any

def create_train_state(model: nn.Module, key, dummy_data):
    params = model.init(key, dummy_data)['params']
    # TrainState is a simple built-in wrapper class that makes things a bit cleaner
    return train_state.TrainState.create(apply_fn=model.apply, params=params)

def compute_metrics(logits, gt_labels):
    log_softmax = jax.nn.log_softmax(logits, axis=-1)
    loss = -jnp.mean(jnp.sum(log_softmax * jax.nn.one_hot(gt_labels, logits.shape[-1]), axis=-1))
    accuracy = jnp.mean(jnp.argmax(logits, -1) == gt_labels)

    metrics = {
        'loss': loss,
        'accuracy': accuracy,
    }
    return metrics

@partial(jax.jit, static_argnames=("n_inner_gradient_steps",))
def train_step(train_state: train_state.TrainState, train_imgs, train_lbls, test_imgs, test_lbls, n_inner_gradient_steps):
    def loss_fn(params, imgs, lbls):
        logits = train_state.apply_fn({'params': params}, imgs)
        one_hot_gt_labels = jax.nn.one_hot(gt_labels, num_classes=10)
        loss = -jnp.mean(jnp.sum(one_hot_gt_labels * logits, axis=-1))
        return loss, logits
  
    (_, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)  # this is the whole update now! concise!
    metrics = compute_metrics(logits=logits, gt_labels=gt_labels)  # duplicating loss calculation but it's a bit cleaner
    return state, metrics