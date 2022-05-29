# LightningTrainable
## Before all
Tune support a trianable class api, while `tune.run` support a class input rather than a instance input(https://docs.ray.io/en/latest/tune/api_docs/trainable.html#trainable-class-api). So in this demo, I still write a `LightningTrainable` to integrate nano into ray tune.

A patch (`ray_trial_executor.patch`) is involved to set those env var(without KMP_AFFINITY) to each actor.

## API design
This is really an easy POC API design.
```python
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
```

## Example
`model_lightning.py` contains a lightning model taken from an example in ray (https://docs.ray.io/en/releases-1.11.0/tune/tutorials/tune-pytorch-lightning.html)

`lightning_trainable.py` contains the definition of `LightningTrainable` and a main program.
```bash
python lightning_trainable.py
```
