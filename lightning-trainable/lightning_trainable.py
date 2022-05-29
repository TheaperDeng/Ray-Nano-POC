import os
import ray
from ray import tune

from ray.tune.schedulers import ASHAScheduler
from model_lightning import LightningMNISTClassifier

class LightningTrainable(tune.Trainable):
    '''
    This is a class for pytorch-lightning users to transform
    their lightning model to a trainable class.
    This creator will internally using an accelerated pytorch lightning
    trainer(bigdl.nano.pytorch.Trainer).

    Users need to implement a `create_model` method in minimum

    A typical use is:
    >>> class MyTrainable(LightningTrainable):
    >>>     def create_model(self, config):
    >>>         return PL_MODEL(config)
    >>> 
    >>> analysis = tune.run(MyTrainable, ...)
    '''

    # runtime_env for this trainable
    runtime_env = {"env_vars":{"OMP_NUM_THREADS": "3"}}

    def create_model(self, config):
        raise NotImplementedError("Users need to implement this method")

    def setup(self, config):
        from bigdl.nano.pytorch import Trainer
        self.model = self.create_model(config=config)
        self.trainer = Trainer(max_epochs=1, use_ipex=True)

    def step(self):
        self.trainer.fit(self.model)
        valid_result = self.trainer.validate(self.model)
        return valid_result[0]

class MyTrainable(LightningTrainable):
    def create_model(self, config):
        return LightningMNISTClassifier(config,
                                    data_dir="/home/junweid/bug-reproduce/ray-lightningtrainable")

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
