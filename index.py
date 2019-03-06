from __future__ import print_function, division, absolute_import
from torchvision import transforms
from torch.autograd import Variable
import torch
import PIL
import numpy as np
import tensorflow as tf
from importlib.machinery import SourceFileLoader

dictTensorflowPretrained = {
    'strTensor' : '', # output tensor name
    'strModelPath' : '', # *.tfmodel file path
    'strModelName' : '', # model name
    'dictModelInput' : { # input dict
    },
    'strModelInputTensor' : '' # input tensor name
}

dictTensorflowCheckpoint = {
    'strTensor' : '', # output tensor name
    'strModelMetaPath' : '', # *.ckpt.meta file path
    'strModelCheckpointPath' : '', # *.ckpt file path
    'dictModelInput': { # input dict
    },
    'strModelInputTensor' : ''# input tensor name
}

dictPytorch = {
    'strModelMetaPath' : '', # *.py file path
    'strModelCheckpointPath':'' # *.pth file path
}

vecPaths = []

def runTensorflowPretrained(npimage,strTensor,strModelPath,strModelName,dictModelInput,strModelInputTensor):
    with tf.gfile.FastGFile(strModelPath, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name=strModelName)
    g = tf.get_default_graph()
    tensor = g.get_tensor_by_name(strTensor)
    sess = tf.InteractiveSession()
    dictModelInput[strModelInputTensor] = npimage
    outTF = np.squeeze(tensor.eval(feed_dict=dictModelInput))
    sess.close()
    return outTF

def runTensorflowCheckpoint(npimage,strTensor,strModelMetaPath,strModelCheckpointPath,dictModelInput,strModelInputTensor):
    sess = tf.InteractiveSession()
    saver = tf.train.import_meta_graph(strModelMetaPath)
    saver.restore(sess, strModelCheckpointPath)
    g = tf.get_default_graph()
    tensor = g.get_tensor_by_name(strTensor)
    dictModelInput[strModelInputTensor] = npimage
    outTFC = np.squeeze(tensor.eval(feed_dict=dictModelInput))
    sess.close()
    return outTFC

def runPytrochCheckpoint(torchimage,strModelMetaPath,strModelCheckpointPath):
    SourceFileLoader('MainModel', strModelMetaPath).load_module()
    the_model = torch.load(strModelCheckpointPath)
    the_model.eval()

    outTorch = the_model(torchimage).data.cpu().numpy()
    return outTorch


def preprocessImage(imagePath):
    image = PIL.Image.open(imagePath).convert('RGB')

    normalize = transforms.Normalize(
       mean=[0.485, 0.456, 0.406],
       std=[0.0039215, 0.0039215, 0.0039215]
    )

    test_transformsC = transforms.Compose([
                                        transforms.Resize((224, 224)),
                                          transforms.ToTensor(),
                                          normalize
                                         ])
    test_transformsTF = transforms.Compose([
                                            transforms.Resize((224, 224)),
                                          transforms.ToTensor()
                                         ])
    image_tensor = test_transformsTF(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    torchImage = Variable(image_tensor)
    npImageTF = np.array(torchImage).swapaxes(1, 3).swapaxes(1, 2)

    image_tensor = test_transformsC(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    torchImage = Variable(image_tensor)
    npImageC = np.array(torchImage).swapaxes(1, 3).swapaxes(1, 2)

    return (npImageTF,npImageC,torchImage)

def preprocessImages(vecPath):
    normalize = transforms.Normalize(
       mean=[0.485, 0.456, 0.406],
       std=[0.0039215, 0.0039215, 0.0039215]
    )

    test_transformsC = transforms.Compose([
                                        transforms.Resize((224, 224)),
                                          transforms.ToTensor(),
                                          normalize
                                         ])
    test_transformsTF = transforms.Compose([
                                            transforms.Resize((224, 224)),
                                          transforms.ToTensor()
                                         ])
    vecTensors = []
    for imagePath in vecPath:
        image = PIL.Image.open(imagePath).convert('RGB')
        image_tensor = test_transformsTF(image).float()
        image_tensor = image_tensor.unsqueeze_(0)
        vecTensors.append(image_tensor)
    torchImages = Variable(torch.cat(vecTensors,0))
    npImageTF = np.array(torchImages).swapaxes(1, 3).swapaxes(1, 2)

    vecTensors = []
    for imagePath in vecPath:
        image = PIL.Image.open(imagePath).convert('RGB')
        image_tensor = test_transformsC(image).float()
        image_tensor = image_tensor.unsqueeze_(0)
        vecTensors.append(image_tensor)
    torchImages = Variable(torch.cat(vecTensors, 0))
    npImageC = np.array(torchImages).swapaxes(1, 3).swapaxes(1, 2)

    return (npImageTF,npImageC,torchImages)

def runComparisonImage(strPath,dictTFPretrained,dictTFCheckpoint,dictTorch):
    (npImagePretrained,npImageCheckpoint,torchImage) = preprocessImage(strPath)

    outTFpretrained = runTensorflowPretrained(npImagePretrained,
                                         dictTFPretrained['strTensor'],
                                         dictTFPretrained['strModelPath'],
                                         dictTFPretrained['strModelName'],
                                         dictTFPretrained['dictModelInput'],
                                         dictTFPretrained['strModelInputTensor'])
    outTFcheckpoint = runTensorflowCheckpoint(npImageCheckpoint,
                                         dictTFCheckpoint['strTensor'],
                                         dictTFCheckpoint['strModelMetaPath'],
                                         dictTFCheckpoint['strModelCheckpointPath'],
                                         dictTFCheckpoint['dictModelInput'],
                                         dictTFCheckpoint['strModelInputTensor'])
    outTorch = runPytrochCheckpoint(torchImage,
                               dictTorch['strModelMetaPath'],
                               dictTorch['strModelCheckpointPath'])

    outTFpretrainedNorm = (outTFpretrained - np.mean(outTFpretrained))/np.std(outTFpretrained)
    outTFcheckpointNorm = (outTFcheckpoint - np.mean(outTFcheckpoint))/np.std(outTFcheckpoint)
    outTorchNorm = (outTorch - np.mean(outTorch))/np.std(outTorch)
    return (np.sum(np.abs(outTFpretrainedNorm-outTorchNorm)),
            np.mean(np.abs(outTFpretrainedNorm - outTorchNorm)),
            np.median(np.abs(outTFpretrainedNorm - outTorchNorm)))

def calcNorm(outModel):
    outModelMean = np.repeat(np.mean(outModel,axis=1)[...,np.newaxis], outModel.shape[1], 1)
    outModelStd = np.repeat(np.std(outModel, axis=1)[...,np.newaxis], outModel.shape[1], 1)
    return (outModel-outModelMean)/outModelStd

def runComparisonImages(vecPaths,dictTFPretrained,dictTFCheckpoint,dictTorch):
    (npImagePretrained, npImageCheckpoint, torchImage) = preprocessImages(vecPaths)

    outTFpretrained = runTensorflowPretrained(npImagePretrained,
                                         dictTFPretrained['strTensor'],
                                         dictTFPretrained['strModelPath'],
                                         dictTFPretrained['strModelName'],
                                         dictTFPretrained['dictModelInput'],
                                         dictTFPretrained['strModelInputTensor'])
    outTFcheckpoint = runTensorflowCheckpoint(npImageCheckpoint,
                                         dictTFCheckpoint['strTensor'],
                                         dictTFCheckpoint['strModelMetaPath'],
                                         dictTFCheckpoint['strModelCheckpointPath'],
                                         dictTFCheckpoint['dictModelInput'],
                                         dictTFCheckpoint['strModelInputTensor'])
    outTorch = runPytrochCheckpoint(torchImage,
                               dictTorch['strModelMetaPath'],
                               dictTorch['strModelCheckpointPath'])

    outTFpretrainedNorm = calcNorm(outTFpretrained)
    outTFcheckpointNorm = calcNorm(outTFcheckpoint)
    outTorchNorm = calcNorm(outTorch)

    return (np.sum(np.abs(outTFpretrainedNorm - outTorchNorm),axis=1),
            np.mean(np.abs(outTFpretrainedNorm - outTorchNorm),axis=1),
            np.median(np.abs(outTFpretrainedNorm - outTorchNorm),axis=1))


(vecSum, vecMean, vecMedian) = runComparisonImages(vecPaths,dictTensorflowPretrained,dictTensorflowCheckpoint,dictPytorch)
print(np.mean(vecSum))
print(np.mean(vecMean))
print(np.mean(vecMedian))
print(np.std(vecSum))
print(np.std(vecMean))
print(np.std(vecMedian))