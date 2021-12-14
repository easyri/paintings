from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import argparse
from random import shuffle, seed
import string
# non-standard dependencies:
import h5py
from six.moves import cPickle
import numpy as np
import torch
import torchvision.models as models
import skimage.io
import gc

from torchvision import transforms as trn

preprocess = trn.Compose([
        #trn.ToTensor(),
        trn.Resize(256),
        trn.CenterCrop(224),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

from resnet_utils import myResnet
import resnet as resnet

def main(params):
  net = getattr(resnet, params['model'])()
  net.load_state_dict(torch.load(os.path.join(params['model_root'],params['model']+'.pth')))
  torch.cuda.empty_cache()
  gc.collect()
  my_resnet = myResnet(net)
  my_resnet.cuda()
  my_resnet.eval()

  imgs = json.load(open(params['input_json'], 'r'))
  imgs = imgs['images']
  N = len(imgs)

  seed(123) # make reproducible

  dir_fc = params['output_dir']+'_fc'
  dir_att = params['output_dir']+'_att'
  if not os.path.isdir(dir_fc):
    os.mkdir(dir_fc)
  if not os.path.isdir(dir_att):
    os.mkdir(dir_att)
  for i, img in enumerate(imgs):
      gc.collect()
      torch.cuda.empty_cache()
      # load the image
      I = skimage.io.imread(os.path.join(params['images_root'], img['filepath'], img['filename']))
      # handle grayscale input images
      if len(I.shape) == 2:
        I = I[:,:,np.newaxis]
        I = np.concatenate((I,I,I), axis=2)
      I = I.astype('float32')/255.0
      I = torch.from_numpy(I.transpose([2,0,1])).cuda()
      I = preprocess(I)
      with torch.no_grad():
        tmp_fc, tmp_att = my_resnet(I, params['att_size'])
      # write to pkl
      np.save(os.path.join(dir_fc, str(img['imgid'])), tmp_fc.data.cpu().float().numpy())
      np.savez_compressed(os.path.join(dir_att, str(img['imgid'])), feat=tmp_att.data.cpu().float().numpy())
      del tmp_fc, tmp_att, I
      if i % 100 == 0:
        torch.cuda.empty_cache()
      if i % 1000 == 0:
        print('processing %d/%d (%.2f%% done)' % (i, N, i*100.0/N))

  print('wrote ', params['output_dir'])


if __name__ == "__main__":
  pars = {"input_json": "wikijson.json", "output_dir": "D:\\paintings\\resnet features", "images_root": '',
          'att_size': 7, "model": 'resnet101', 'model_root': ''}
  torch.cuda.empty_cache()
  for obj in gc.get_objects():
    try:
      if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
        print(type(obj), obj.size())
    except:
      pass
  main(pars)
