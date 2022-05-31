import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler

from model_lightning import LightningMNISTClassifier
from lightning_trainable import LightningTrainable


class MyTrainable(LightningTrainable):

    def create_model(self, config):
        return LightningMNISTClassifier(config,
                                        data_dir="/home/junweid/bug-reproduce/ray-lightningtrainable")
    
    def configure_trainer(self):
        return {"max_epochs": 2}

if __name__ == "__main__":
    ray.init(num_cpus=6)
    config = {
        "layer_1_size": tune.choice([32, 64, 128]),
        "layer_2_size": tune.choice([64, 128, 256]),
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([32, 64, 128]),
    }

    sched = ASHAScheduler()
    analysis = tune.run(
        MyTrainable,
        metric="ptl/val_accuracy",
        mode="max",
        scheduler=sched,
        stop={
            "ptl/val_accuracy": 0.95,
            "training_iteration": 2,
        },
        resources_per_trial={"cpu": 3},
        num_samples=2,
        # checkpoint_at_end=True,
        # checkpoint_freq=3,
        config=config,
    )

    print("Best config is:", analysis.best_config)