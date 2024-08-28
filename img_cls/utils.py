from functools import partial
import jax
from flax.training import train_state
from typing import Any
import numpy as np
import optax
import jax.numpy as jnp
import flax.linen as nn

class TrainStateWithBatchNorm(train_state.TrainState):
    batch_stats: Any

def get_optimizer(learning_rate):
    return optax.adam(learning_rate=learning_rate)
    # return optax.sgd(learning_rate=learning_rate)

def get_metrics(step_metrics):
    step_metrics = [jax.device_get(metric) for metric in step_metrics]  # pull from the accelerator onto host (CPU)
    mean_metrics = {k: np.mean([metric[k] for metric in step_metrics]).item() for k in step_metrics[0]}  # mean over meta-batch
    return mean_metrics

def create_train_state(model: nn.Module, key, dummy_data, lr):
    params = model.init(key, dummy_data)['params']
    tx = optax.adam(learning_rate=lr)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

def update_params_naive(params, grads, lr):
    return jax.tree_map(lambda p, g: p - lr * g, params, grads)

def update_params(params, grads, opt, opt_state):
    updates, new_opt_state = opt.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state

def compute_metrics(logits, gt_labels):
    one_hot_gt_labels = jax.nn.one_hot(gt_labels, num_classes=10)
    loss = -jnp.mean(jnp.sum(one_hot_gt_labels * logits, axis=-1))
    accuracy = jnp.mean(jnp.argmax(logits, -1) == gt_labels)

    metrics = {
        'loss': loss,
        'accuracy': accuracy,
    }
    return metrics

@jax.jit
def train_step(train_state: train_state.TrainState, train_task_info):
    def loss_fn(params, imgs, lbls):
        logits = train_state.apply_fn({'params': params}, imgs)
        one_hot_gt_labels = jax.nn.one_hot(lbls, num_classes=10)
        loss = -jnp.mean(jnp.sum(one_hot_gt_labels * logits, axis=-1))
        return loss, logits
    train_images, train_labels = train_task_info
    (_, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(train_state.params, train_images, train_labels)
    train_state = train_state.apply_gradients(grads=grads)
    metrics = compute_metrics(logits=logits, gt_labels=train_labels)
    return train_state, metrics

@jax.jit
def val_step(train_state: train_state.TrainState, val_task_info):
    def loss_fn(params, imgs, lbls):
        logits = train_state.apply_fn({'params': params}, imgs)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=lbls).mean()
        return loss, logits
    val_images, val_labels = val_task_info
    _, val_logits = loss_fn(train_state.params, val_images, val_labels)

    metrics = compute_metrics(logits=val_logits, gt_labels=val_labels)
    return metrics