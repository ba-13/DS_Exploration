## batches.py

> some available datasets by default are torchvision.datasets.MNIST(), fashion-mnist, cifar, coco

While using multiple cores via `num_workers` in `Dataloader` class, you may face an error:

```
An attempt has been made to start a new process before the current process has finished its bootstrapping phase.

This probably means that you are not using fork to start your
child processes and you have forgotten to use the proper idiom
in the main module
...
The above exception was the direct cause of the following exception:
...
RuntimeError: DataLoader worker (pid(s) 4812, 9972) exited unexpectedly
```

If the first exception is not faced, you have to make `num_workers` to a lower value till error is handled.
If you are facing the same error, that's due to recursive calling of the script by the sub-processes created, you can handle this error by including the `Dataloader` in `if __name__ == "__main__":`.

## transform.py

[Transform](https://pytorch.org/docs/stable/torchvision/transforms.html) can be applied to PIL images, tensors, ndarrays, custom data during creation of DataSet.

| Base    | Transforms                                                                                                |
| ------- | --------------------------------------------------------------------------------------------------------- |
| Images  | CenterCrop, Grayscale, Pad, RandomAffine, RandomCrop, RandomHorizontalFlip, RandomRotation, Resize, Scale |
| Tensors | LinearTransformation, Normalize, RandomErasing                                                            |

Conversion :

- ToPILImage : from tensor or ndarray
- ToTensor : from numpy.ndarray

Generic :

- Use Lambda
- Custom Class
- composed = transforms.Compose(Rescale(256), RandomCrop(224)) # multiple transforms
