# LightningTrainable
## Before all
Tune support a trianable class api, while `tune.run` support a class input rather than a instance input(https://docs.ray.io/en/latest/tune/api_docs/trainable.html#trainable-class-api). So in this demo, I still write a `LightningTrainableCreator` to integrate nano into ray tune.

## API design
This is really an easy POC API design.
```python
def LightningTrainableCreator(model_creator):
    '''
    This is a creator for pytorch-lightning users to transform
    their lightning model to a trainable class.
    This creator will internally using an accelerated pytorch lightning
    trainer(bigdl.nano.pytorch.Trainer).

    :param model_creator: a function that could take a config dict and
           return a lightning module instance.

    A typical use is:
    >>> trainable_class = LightningTrainableCreator(model_creator)
    >>> analysis = tune.run(trainable_class, ...)
    '''
```

## Example
`model_lightning.py` contains a lightning model taken from an example in ray (https://docs.ray.io/en/releases-1.11.0/tune/tutorials/tune-pytorch-lightning.html)

`lightning_trainable.py` contains the definition of `LightningTrainableCreator` and a main program.
```bash
python lightning_trainable.py
```

## Known issue
- We can't make the environment variable effective by writing it in `step()` function of trainable class. Possible solution is to ask Ray expose an API in placement group or trainable.