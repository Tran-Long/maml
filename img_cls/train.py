from datetime import datetime
import os
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import jax
import jax.numpy as jnp
import yaml
from data import CIFAR10Dataset, collate_fn
from cnn import CNN
from utils import create_train_state, train_step, val_step, get_metrics
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

rng = jax.random.PRNGKey(37)
TRAIN_CONFIG = yaml.full_load(open(str(Path(__file__).parent / "train_config.yaml"), "r"))
# root_log_dir = TRAIN_CONFIG["root_log_dir"]
# writer = SummaryWriter(os.path.join(root_log_dir, datetime.now().strftime("%Y%m%d-%H%M%S")))


train_dataset = CIFAR10Dataset(mode="train")
test_dataset = CIFAR10Dataset(mode="val")
train_loader = DataLoader(train_dataset, batch_size=TRAIN_CONFIG["batch_size"], shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=TRAIN_CONFIG["batch_size"], shuffle=False, collate_fn=collate_fn)
print(len(train_loader), len(test_loader))

model = CNN(n_classes=10)
# rng, init_key = jax.random.split(rng)
dummy_data = jnp.ones([1, *TRAIN_CONFIG["input_shape"]])
train_state = create_train_state(model, rng, dummy_data, TRAIN_CONFIG["lr"])


for epoch in range(TRAIN_CONFIG["n_epochs"]):
    epoch_metrics = []
    for step, train_batch in enumerate(train_loader):
        train_state, train_metrics = train_step(train_state, train_batch)
        epoch_metrics.append(train_metrics)
    epoch_metrics = get_metrics(epoch_metrics)
    print(f"Epoch {epoch}: Train Loss: {epoch_metrics['loss']:.4f}, Train Accuracy: {epoch_metrics['accuracy']:.4f}")
    # writer.add_scalar("Loss/train", train_metrics["loss"], step)
    # writer.add_scalar("Accuracy/train", train_metrics["accuracy"], step)
    
    all_val_metrics = []
    for val_batch in test_loader:
        val_metrics = val_step(train_state, val_batch)
        all_val_metrics.append(val_metrics)
    all_val_metrics = get_metrics(all_val_metrics)
#     writer.add_scalar("Loss/val", val_metrics["loss"], step)
#     writer.add_scalar("Accuracy/val", val_metrics["accuracy"], step)
    print(f"Val Loss: {all_val_metrics['loss']:.4f}, Val Accuracy: {all_val_metrics['accuracy']:.4f}")

