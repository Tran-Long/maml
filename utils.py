from functools import partial
import jax
from flax.training import train_state
from typing import Any
import numpy as np
import optax
import jax.numpy as jnp
import flax.linen as nn
from data.data import MetaDataset

def get_optimizer(learning_rate):
    return optax.adam(learning_rate=learning_rate)
    # return optax.sgd(learning_rate=learning_rate)

def get_metrics(step_metrics):
    step_metrics = [jax.device_get(metric) for metric in step_metrics]  # pull from the accelerator onto host (CPU)
    mean_metrics = {k: np.mean([metric[k] for metric in step_metrics]).item() for k in step_metrics[0]}  # mean over meta-batch
    return mean_metrics

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

def compute_metrics(logits, gt_labels, additional_info={}):
    one_hot_gt_labels = jax.nn.one_hot(gt_labels, num_classes=logits.shape[-1])
    loss = -jnp.mean(jnp.sum(one_hot_gt_labels * logits, axis=-1))
    accuracy = jnp.mean(jnp.argmax(logits, -1) == gt_labels)

    metrics = {
        'loss': loss,
        'accuracy': accuracy,
        **additional_info
    }
    return metrics

@partial(jax.jit, static_argnums=(2, ))
def train_step(state: train_state.TrainState, train_task_info, n_inner_gradient_steps, alpha):
    def loss_fn(params, imgs, lbls):
        logits = state.apply_fn({'params': params}, imgs)
        one_hot_gt_labels = jax.nn.one_hot(lbls, num_classes=logits.shape[-1])
        loss = -jnp.mean(jnp.sum(one_hot_gt_labels * logits, axis=-1))
        return loss, logits

    def meta_loss_fn(params, train_imgs, train_lbls, test_imgs, test_lbls):
        # inner_opt = optax.adam(learning_rate=alpha)
        # inner_opt_state = inner_opt.init(params)
        for _ in range(n_inner_gradient_steps):
            grads, _ = jax.grad(loss_fn, has_aux=True)(params, train_imgs, train_lbls)
            params = jax.tree_map(lambda p, g: p - alpha * g, params, grads)
            # updates, inner_opt_state = inner_opt.update(grads, inner_opt_state, params)
            # params = optax.apply_updates(params, updates)
        return loss_fn(params, test_imgs, test_lbls)

    train_images, train_labels, test_images, test_labels = train_task_info
    step_metrics = []
    params = state.params.copy()
    for train_imgs, train_lbls, test_imgs, test_lbls in zip(train_images, train_labels, test_images, test_labels):
        meta_grad, test_logits = jax.grad(meta_loss_fn, has_aux=True)(params, train_imgs, train_lbls, test_imgs, test_lbls)     
        state = state.apply_gradients(grads=meta_grad)
        metrics = compute_metrics(logits=test_logits, gt_labels=test_lbls)
        step_metrics.append(metrics)
    return state, step_metrics

@partial(jax.jit, static_argnums=(2,))
def val_step(state: train_state.TrainState, val_task_info, n_finetune_gradient_steps, alpha):
    def loss_fn(params, imgs, lbls):
        logits = state.apply_fn({'params': params}, imgs)
        one_hot_gt_labels = jax.nn.one_hot(lbls, num_classes=logits.shape[-1])
        loss = -jnp.mean(jnp.sum(one_hot_gt_labels * logits, axis=-1))
        return loss, logits
    train_images, train_labels, test_images, test_labels = val_task_info
    step_metrics = []
    meta_params = state.params.copy()
    for train_imgs, train_lbls, test_imgs, test_lbls in zip(train_images, train_labels, test_images, test_labels):
        params = meta_params.copy()
        # inner_opt = optax.adam(learning_rate=0.001)
        # inner_opt_state = inner_opt.init(params)
        for _ in range(n_finetune_gradient_steps):
            grads, _ = jax.grad(loss_fn, has_aux=True)(params, train_imgs, train_lbls)
            params = jax.tree_map(lambda p, g: p - alpha * g, params, grads)
            # updates, inner_opt_state = inner_opt.update(grads, inner_opt_state, params)
            # params = optax.apply_updates(params, updates)

        _, test_logits = loss_fn(params, test_imgs, test_lbls)
        metrics = compute_metrics(logits=test_logits, gt_labels=test_lbls)
        step_metrics.append(metrics)
    return step_metrics 