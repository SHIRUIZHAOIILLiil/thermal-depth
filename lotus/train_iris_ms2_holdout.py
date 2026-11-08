"""train_iris_ms2_g.py with HoldoutDataset; every other line of training is the original."""
import train_iris_ms2_g as trainer
from utils.ms2_holdout import HoldoutDataset

trainer.MS2ThermalDataset = HoldoutDataset

if __name__ == "__main__":
    trainer.main()
