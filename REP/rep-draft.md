## Summary

### General Motivation

motivation1: Many users of Ray Tune is working on the HPO (hyperparameter optimization) of their pytorch (including pytorch-lightning) model and keras model. Currently users need to write a trainable function / trainable class with some boilerplate code. It can be useful to provide a high-level but flexible API(`LightningTrainable`, `KerasTrainable`, `TorchTrainable`) to ease the usage so that users can only provide a model creator, a data creator and a search space to complete the tuning.

*motivation2*: It's highly possible that users using a cluster with many CPU (cores). CPU training is a valid choice for AI models' training, while currently CPU training acceleration is not applied (and hard to apply) in ray tune. [bigdl-nano](https://github.com/intel-analytics/BigDL/tree/main/python/nano) is a python package to transparently accelerate Pytorch and Tensorflow applications on Intel hardware (CPU for now). For Pytorch, [bigdl-nano](https://github.com/intel-analytics/BigDL/tree/main/python/nano) provides an extended version of PyTorch-Lightning `Trainer`. Optimization such as Best Known Configurations (malloc, omp settings, ...), ipex(intel pytorch extension) can be enabled automatically when users do not request GPU.

### Should this change be within `ray` or outside?

This enhancment should be with in `ray.tune`. We could add three trainable class (`LightningTrainable`, `KerasTrainable`, `TorchTrainable`) in several phases.

## Design and Architecture

### LightningTrainable API

Start from `LightningTrainable`. 

```python
class LightningTrainable(tune.Trainable):
    '''
    This is a class for pytorch-lightning users to transform
    their lightning model to a trainable class.
    This creator will internally using an accelerated pytorch lightning
    trainer(bigdl.nano.pytorch.Trainer).

    Users need to implement a `create_model` method in minimum
    '''

    def create_model(self, config):
        '''
        User should always overwrite this method
        `create_model` takes a config dictionary and returns a pytorch lightning module.

        :param config: the config dictionary for model creator.

        :return: a pytorch-lightning model instance.
        '''

    def configrate_trainer(self):
        '''
        User can optionally overwrite this method.
        A default trainer setting will be {"max_epochs": 1}
        `configrate_trainer` returns a dictionary to pytorch-lightining Trainer.
        Users may refer to https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html

        :return: a dictionary to be passed to pytorch-lightning trainer.
        '''
```

This `LightningTrainable` can be used by customers easily through

```python
class MyTrainable(LightningTrainable):

    def create_model(self, config):
        return PL_MODEL(config)

    def configrate_trainer(self):
        return {"max_epochs": 2}

analysis = tune.run(MyTrainable, ...)
```

### Details of `LightningTrainable`

`LightningTrainable` will implement `step` and `setup` for users.

In `setup`, a model instance and a trainer instance will be created. The trainer will be optimized according to the resource requested by the user(if using GPU, original pl trainer is used. if using CPU, a bigdl-nano pl trainer will be used).

In `step`, the model will be fitted by the trainer we just created in `setup`. `trainer.validate` will be called for validation and returned as the validation result.



Users can overwrite `create_model` and `configrate_trainer`.

Users should always overwrite `create_model`. It takes a config dictionary and returns a pytorch lightning module instance.

User can optionally overwrite `configrate_trainer`. It returns a dictionary to initialize a pytorch-lightining Trainer. e.g. Users may return `{"max_epochs": 2}`, so that 2 epochs will be trained for each `training_iteration`.



### runtime_env for trainable

users' customized runtime_env is important for training performance improvement and, to be more generally, requested by some users https://github.com/ray-project/ray/issues/23234 . In our proposal, we proposed an interface in `tune.trainable` for users to set runtime env customizedly

[NOT COMPLETED]





A prototype code and example is stated in https://github.com/TheaperDeng/Ray-Nano-POC/tree/main/lightning-trainable