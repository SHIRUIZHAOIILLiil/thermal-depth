"""train_iris_ms2_g.py with StrongTransform; every other line of training is the original."""
import train_iris_ms2_g as trainer
from utils.ms2_train_augment import StrongTransform

trainer.MS2ThermalTransform = StrongTransform

if __name__ == "__main__":
    trainer.main()
