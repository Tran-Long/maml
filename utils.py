from functools import partial
import jax
from flax.training import train_state
from typing import Any
import optax
import jax.numpy as jnp
import flax.linen as nn

class TrainStateWithBatchNorm(train_state.TrainState):
    batch_stats: Any

def get_optimizer(learning_rate):
    # return optax.adam(learning_rate=learning_rate)
    return optax.sgd(learning_rate=learning_rate)

def get_metrics(metrics):
    metrics = jax.device_get(metrics)  # pull from the accelerator onto host (CPU)
    metrics = jax.tree_map(lambda x: x.item(), metrics)  # np.ndarray -> scalar
    return metrics

def create_train_state(model: nn.Module, key, dummy_data, beta):
    params = model.init(key, dummy_data)['params']
    tx = get_optimizer(beta)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

def update_params_naive(params, grads, lr):
    return jax.tree_map(lambda p, g: p - lr * g, params, grads)

def update_params(params, grads, opt, opt_state):
    updates, new_opt_state = opt.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state

def compute_metrics(logits, gt_labels):
    loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=gt_labels).mean()
    accuracy = jnp.mean(jnp.argmax(logits, -1) == gt_labels)

    metrics = {
        'loss': loss,
        'accuracy': accuracy,
    }
    return metrics

@partial(jax.jit, static_argnums=(2))
def train_step(train_state: train_state.TrainState, task_info, n_inner_gradient_steps, alpha):
    def loss_fn(params, imgs, lbls):
        logits = train_state.apply_fn({'params': params}, imgs)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=lbls).mean()
        return loss, logits
    
    train_imgs, train_lbls, test_imgs, test_lbls = task_info
    params = train_state.params
    inner_opt = get_optimizer(alpha)
    inner_opt_state = inner_opt.init(params)
    for _ in range(n_inner_gradient_steps):
        grads, _ = jax.grad(loss_fn, has_aux=True)(params, train_imgs, train_lbls)
        updates, inner_opt_state = inner_opt.update(grads, inner_opt_state, params)
        params = optax.apply_updates(params, updates)

    grads, test_logits = jax.grad(loss_fn, has_aux=True)(params, test_imgs, test_lbls)

    train_state.apply_gradients(grads=grads)
    metrics = compute_metrics(logits=test_logits, gt_labels=test_lbls)  # duplicating loss calculation but it's a bit cleaner
    return train_state, metrics

@partial(jax.jit, static_argnames=("n_finetune_gradient_steps",))
def val_step(train_state: train_state.TrainState, task_info, n_finetune_gradient_steps, alpha):
    def loss_fn(params, imgs, lbls):
        logits = train_state.apply_fn({'params': params}, imgs)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=lbls).mean()
        return loss, logits
    
    train_imgs, train_lbls, test_imgs, test_lbls = task_info
    params = train_state.params
    inner_opt = get_optimizer(alpha)
    inner_opt_state = inner_opt.init(params)
    for _ in range(n_finetune_gradient_steps):
        grads, _ = jax.grad(loss_fn, has_aux=True)(params, train_imgs, train_lbls)
        updates, inner_opt_state = inner_opt.update(grads, inner_opt_state, params)
        params = optax.apply_updates(params, updates)

    _, test_logits = loss_fn(params, test_imgs, test_lbls)
    metrics = compute_metrics(logits=test_logits, gt_labels=test_lbls)
    return metrics