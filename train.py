from datetime import datetime
import os
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import jax
import jax.numpy as jnp
import yaml
from data.data import MetaDataset
from model.cnn import CNN
from utils import create_train_state, train_step, val_step, get_metrics
from torch.utils.tensorboard import SummaryWriter

rng = jax.random.PRNGKey(37)
TRAIN_CONFIG = yaml.full_load(open(str(Path(__file__).parent / "train_config.yaml"), "r"))
root_log_dir = TRAIN_CONFIG["root_log_dir"]
writer = SummaryWriter(os.path.join(root_log_dir, datetime.now().strftime("%Y%m%d-%H%M%S")))

alpha, beta = TRAIN_CONFIG["alpha"], TRAIN_CONFIG["beta"]
n_inner_gradient_steps = TRAIN_CONFIG["n_inner_gradient_steps"]
n_finetune_gradient_steps = TRAIN_CONFIG["n_finetune_gradient_steps"]
meta_batch_size = TRAIN_CONFIG["meta_batch_size"]
meta_batch_size_eval = TRAIN_CONFIG.get("meta_batch_size_eval", meta_batch_size)

meta_train_dataset = MetaDataset(mode="train")
meta_test_dataset = MetaDataset(mode="val")

model = CNN()
rng, init_key = jax.random.split(rng)
dummy_data = jnp.ones([1, *TRAIN_CONFIG["input_shape"]])
train_state = create_train_state(model, init_key, dummy_data, beta)

for step in range(TRAIN_CONFIG["n_steps"]):
    for _ in range(meta_batch_size):
        train_dataset, test_dataset = meta_train_dataset.sample_task()
        train_task_info = [*train_dataset.sample(), *test_dataset.sample()]
        train_state, train_metrics = train_step(train_state, train_task_info, n_inner_gradient_steps, meta_batch_size, alpha)
        train_metrics = get_metrics(train_metrics)
        print(f"Step {step}: Train Loss: {train_metrics['loss']:.4f}, Val Accuracy: {train_metrics['accuracy']:.4f}")
        writer.add_scalar("Loss/train", train_metrics["loss"], step)
        writer.add_scalar("Accuracy/train", train_metrics["accuracy"], step)
    if step % TRAIN_CONFIG["val_interval"] == 0:
        train_dataset, test_dataset = meta_test_dataset.sample_task()
        val_task_info = [*train_dataset.sample(), *test_dataset.sample()]
        val_metrics = val_step(train_state, val_task_info, n_finetune_gradient_steps, meta_batch_size_eval, alpha)
        val_metrics = get_metrics(val_metrics)
        writer.add_scalar("Loss/val", val_metrics["loss"], step)
        writer.add_scalar("Accuracy/val", val_metrics["accuracy"], step)
        print(f"Step {step}: Val Loss: {val_metrics['loss']:.4f}, Val Accuracy: {val_metrics['accuracy']:.4f}")

