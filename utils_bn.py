from functools import partial
import jax
from flax.training import train_state
from typing import Any
import numpy as np
import optax
import jax.numpy as jnp
import flax.linen as nn
from data.data import MetaDataset

class TrainStateWithBatchNorm(train_state.TrainState):
    batch_stats: Any

def get_optimizer(learning_rate):
    return optax.adam(learning_rate=learning_rate)
    # return optax.sgd(learning_rate=learning_rate)

def get_metrics(step_metrics):
    step_metrics = [jax.device_get(metric) for metric in step_metrics]  # pull from the accelerator onto host (CPU)
    mean_metrics = {k: np.mean([metric[k] for metric in step_metrics]).item() for k in step_metrics[0]}  # mean over meta-batch
    return mean_metrics

def create_train_state(model: nn.Module, key, dummy_data, beta):
    variables = model.init(key, dummy_data, train=False)
    params = variables['params']
    batch_stats = variables['batch_stats']
    tx = get_optimizer(beta)
    return TrainStateWithBatchNorm.create(apply_fn=model.apply, params=params, tx=tx, batch_stats=batch_stats)

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
def train_step(state: TrainStateWithBatchNorm, train_task_info, n_inner_gradient_steps, alpha):
    def loss_fn(params, batch_stats, imgs, lbls):
        logits, updates = state.apply_fn({'params': params, 'batch_stats': batch_stats}, imgs, train=True, mutable=['batch_stats'])
        one_hot_gt_labels = jax.nn.one_hot(lbls, num_classes=logits.shape[-1])
        loss = -jnp.mean(jnp.sum(one_hot_gt_labels * logits, axis=-1))
        return loss, (logits, updates["batch_stats"])

    def meta_loss_fn(params, batch_stats, train_imgs, train_lbls, test_imgs, test_lbls):
        for _ in range(n_inner_gradient_steps):
            grad, (_, batch_stats) = jax.grad(loss_fn, has_aux=True)(params, batch_stats, train_imgs, train_lbls)
            params = jax.tree_map(lambda p, g: p - alpha * g, params, grad)
        test_logits = state.apply_fn({'params': params, 'batch_stats': batch_stats}, test_imgs, train=False)
        one_hot_gt_labels = jax.nn.one_hot(test_lbls, num_classes=test_logits.shape[-1])
        loss = -jnp.mean(jnp.sum(one_hot_gt_labels * test_logits, axis=-1))
        return loss, (test_logits, batch_stats)

    train_images, train_labels, test_images, test_labels = train_task_info
    step_metrics = []
    params = state.params.copy()
    for train_imgs, train_lbls, test_imgs, test_lbls in zip(train_images, train_labels, test_images, test_labels):
        # task_params = params.copy()
        meta_grad, (test_logits, batch_stats) = jax.grad(meta_loss_fn, has_aux=True)(params, state.batch_stats, train_imgs, train_lbls, test_imgs, test_lbls)     
        state = state.apply_gradients(grads=meta_grad)
        state = state.replace(batch_stats=batch_stats)
        metrics = compute_metrics(logits=test_logits, gt_labels=test_lbls)
        step_metrics.append(metrics)
    return state, step_metrics

@partial(jax.jit, static_argnums=(2,))
def val_step(state: TrainStateWithBatchNorm, val_task_info, n_finetune_gradient_steps, alpha):
    def loss_fn(params, batch_stats, imgs, lbls):
        logits, updates = state.apply_fn({'params': params, 'batch_stats': batch_stats}, imgs, train=True, mutable=['batch_stats'])
        one_hot_gt_labels = jax.nn.one_hot(lbls, num_classes=logits.shape[-1])
        loss = -jnp.mean(jnp.sum(one_hot_gt_labels * logits, axis=-1))
        return loss, (logits, updates["batch_stats"])
    train_images, train_labels, test_images, test_labels = val_task_info
    step_metrics = []
    meta_params = state.params.copy()
    meta_batch_stats = state.batch_stats
    for train_imgs, train_lbls, test_imgs, test_lbls in zip(train_images, train_labels, test_images, test_labels):
        params = meta_params.copy()
        batch_stats = meta_batch_stats.copy()
        # inner_opt = get_optimizer(alpha)
        # inner_opt_state = inner_opt.init(params)
        for _ in range(n_finetune_gradient_steps):
            grads, (_, batch_stats) = jax.grad(loss_fn, has_aux=True)(params, batch_stats, train_imgs, train_lbls)
            params = update_params_naive(params, grads, alpha)
            # updates, inner_opt_state = inner_opt.update(grads, inner_opt_state, params)
            # params = optax.apply_updates(params, updates)

        test_logits = state.apply_fn({'params': params, 'batch_stats': batch_stats}, test_imgs, train=False)
        metrics = compute_metrics(logits=test_logits, gt_labels=test_lbls)
        step_metrics.append(metrics)
    return step_metrics