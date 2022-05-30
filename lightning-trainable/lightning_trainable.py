import ray
from ray import tune

class LightningTrainable(tune.Trainable):
    '''
    This class should only be used by being inherited.
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

    def create_model(self, config):
        '''
        User should always overwrite this method

        `create_model` takes a config dictionary and returns a pytorch lightning module.
        '''
        raise NotImplementedError("Users need to implement this method")
    
    def configrate_trainer(self):
        '''
        User can optionally overwrite this method.
        A default trainer setting will be {"max_epochs": 1}

        `configrate_trainer` returns a dictionary to pytorch-lightining Trainer.
        Users may refer to https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html
        '''
        return {"max_epochs": 1}

    def setup(self, config):
        '''
        This method should not be overwritten by users of LightningTrainable

        It create a model instance and a trainer instance.
        Set cpu best known methods in trainer and
        disable cpu best known methods if GPU is requested
        '''
        from bigdl.nano.pytorch import Trainer

        self.model = self.create_model(config=config)
        trainer_config = self.configrate_trainer()

        resources = ray.cluster_resources()
        if "GPU" not in resources:
            # TODO: what if the operations in user's model is not supported by ipex?
            trainer_config.update({"use_ipex": True})
        self.trainer = Trainer(**trainer_config)

    def step(self):
        '''
        This method should not be overwritten by users of LightningTrainable
        '''
        self.trainer.fit(self.model)
        valid_result = self.trainer.validate(self.model)
        return valid_result[0]

    @staticmethod
    def get_runtime_env(placement_group_factory):
        '''
        This method can be defined as an API in `tune.Trainable`
        for users to set runtime environment customizedly.

        This method should not be overwritten by users of LightningTrainable

        Set cpu best known methods in runtime_env and
        disable cpu best known methods if GPU is requested
        '''
        # disable cpu best known methods if GPU is requested
        required_resources = placement_group_factory.required_resources
        if "GPU" in required_resources:
            return {}

        # set cpu best known methods in runtime_env
        # TODO: need a python API for other env variables.
        cpu_num = placement_group_factory.head_cpus
        runtime_env = {"env_vars":{"OMP_NUM_THREADS": str(int(cpu_num))}}
        return runtime_env
