from datetime import datetime
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
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

rng = jax.random.PRNGKey(0)
TRAIN_CONFIG = yaml.full_load(open(str(Path(__file__).parent / "train_config.yaml"), "r"))
root_log_dir = TRAIN_CONFIG["root_log_dir"]
writer = SummaryWriter(os.path.join(root_log_dir, datetime.now().strftime("%Y%m%d-%H%M%S")))

meta_train_dataset = MetaDataset(mode="train")
meta_test_dataset = MetaDataset(mode="val")

alpha, beta = TRAIN_CONFIG["alpha"], TRAIN_CONFIG["beta"]
n_inner_gradient_steps = TRAIN_CONFIG["n_inner_gradient_steps"]
n_finetune_gradient_steps = TRAIN_CONFIG["n_finetune_gradient_steps"]
meta_batch_size = TRAIN_CONFIG["meta_batch_size"]
meta_batch_size_eval = TRAIN_CONFIG.get("meta_batch_size_eval", meta_batch_size)
meta_batch_size_eval = meta_batch_size_eval if isinstance(meta_batch_size_eval, int) else int(meta_batch_size_eval * len(meta_test_dataset))


model = CNN(n_classes=meta_train_dataset.n_ways)
rng, init_key = jax.random.split(rng)
dummy_data = jnp.ones([1, *TRAIN_CONFIG["input_shape"]])
state = create_train_state(model, init_key, dummy_data, beta)

for step in range(TRAIN_CONFIG["n_steps"]):
    train_images, train_labels, test_images, test_labels = [], [], [], []
    for _ in range(meta_batch_size):
        train_dataset, test_dataset = meta_train_dataset.sample_task()
        train_imgs, train_lbls = train_dataset.sample()
        test_imgs, test_lbls = test_dataset.sample()
        train_images.append(train_imgs)
        train_labels.append(train_lbls)
        test_images.append(test_imgs)
        test_labels.append(test_lbls)
    train_task_info = [train_images, train_labels, test_images, test_labels]
    state, train_metrics = train_step(state, train_task_info, n_finetune_gradient_steps, alpha)
    train_metrics = get_metrics(train_metrics)
    print(f"Step {step}: Train Loss: {train_metrics['loss']:.4f}, Train Accuracy: {train_metrics['accuracy']:.4f}")
    # writer.add_scalar("Loss/train", train_metrics["loss"], step)
    # writer.add_scalar("Accuracy/train", train_metrics["accuracy"], step)
    if step % TRAIN_CONFIG["val_interval"] == 0:
        train_images, train_labels, test_images, test_labels = [], [], [], []
        for _ in range(meta_batch_size_eval):
            train_imgs, train_lbls = train_dataset.sample()
            test_imgs, test_lbls = test_dataset.sample()
            train_images.append(train_imgs)
            train_labels.append(train_lbls)
            test_images.append(test_imgs)
            test_labels.append(test_lbls)
        val_task_info = [train_images, train_labels, test_images, test_labels]
        val_metrics = val_step(state, val_task_info, n_finetune_gradient_steps, alpha)
        val_metrics = get_metrics(val_metrics)
        #     writer.add_scalar("Loss/val", val_metrics["loss"], step)
        #     writer.add_scalar("Accuracy/val", val_metrics["accuracy"], step)
        print(f"[VAL] Step {step}: Val Loss: {val_metrics['loss']:.4f}, Val Accuracy: {val_metrics['accuracy']:.4f}")

