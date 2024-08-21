from pathlib import Path
import jax
import jax.numpy as jnp
import yaml
from data.data import MetaDataset
from model.cnn import CNN
from utils import create_train_state

TRAIN_CONFIG = yaml.full_load(open(str(Path(__file__).parent / "train_config.yaml"), "r"))
rng = jax.random.PRNGKey(37)

meta_train_dataset = MetaDataset(mode="train")
meta_test_dataset = MetaDataset(mode="val")

model = CNN()
rng, init_key = jax.random.split(rng)
dummy_data = jnp.ones([1, *TRAIN_CONFIG["input_shape"]])
train_state = create_train_state(model, init_key, dummy_data)

