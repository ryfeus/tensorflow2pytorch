# tensorflow2pytorch
Example which compares pretrained TF model, checkpoint of TF model and pytorch model converted from checkpoint.


## Download and convert model

We will be using MMdnn library (https://github.com/Microsoft/MMdnn) to download and convert model. For example to download and convert VGG19 model you need to do the following:

```bash
mmdownload -f tensorflow -n vgg19
mmconvert -sf tensorflow -in imagenet_vgg19.ckpt.meta -iw imagenet_vgg19.ckpt --dstNodeName MMdnn_Output -df pytorch -om tf_vgg19_to_pth.pth
```

## Compare model results

You can use main.py to compare models' predictions. You will need to keep in mind the following parameters:

Tensorflow pretrained model:

- *.tfmodel file - strModelPath parameter
- model name - strModelName parameter
- name of input tensor - strTensor parameter
- input dict for the model - dictModelInput parameter
- name of output tensor - strModelInputTensor parameter

Tensorflow checkpoint:

- *.ckpt.meta file - strModelMetaPath parameter
- *.ckpt file - strModelCheckpointPath parameter
- name of input tensor - strTensor parameter
- input dict for the model - dictModelInput parameter
- name of output tensor - strModelInputTensor parameter

Pytorch checkpoint:

- *.py file - strModelMetaPath parameter
- *.pth file - strModelCheckpointPath parameter

Input images:

- paths to images - vecPaths parameter