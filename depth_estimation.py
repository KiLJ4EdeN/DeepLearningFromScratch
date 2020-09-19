import PIL.Image as pil
import numpy as np
import mxnet as mx
from mxnet.gluon.data.vision import transforms
import gluoncv
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class DepthEstimator(object):
  def __init__(self, width=640, height=192, ctx=0):
    # resize for model
    self.width = width
    self.height = height
    # original image size
    self.original_width = None
    self.original_height = None
    # utlils
    self.ctx = mx.cpu(ctx)
    self.model = self.load_resnet_monodepth()

  def preprocess_image(self, img):
    self.original_width, self.original_height = img.size
    img = img.resize((self.width, self.height), pil.LANCZOS)
    img = transforms.ToTensor()(mx.nd.array(img)).expand_dims(0).as_in_context(context=self.ctx)
    return img
    
  def predict(self, img):
    img = self.preprocess_image(img)
    outputs = self.model.predict(img)
    disp = outputs[("disp", 0)]
    disp_resized = mx.nd.contrib.BilinearResize2D(disp,
                                                  height=self.original_height,
                                                  width=self.original_width)
    disp_resized_np = disp_resized.squeeze().as_in_context(mx.cpu()).asnumpy()
    vmax = np.percentile(disp_resized_np, 95)
    normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
    colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
    im = pil.fromarray(colormapped_im)
    im.save('test_output.png')
    disp_map = mpimg.imread('test_output.png')
    return disp_map

  def load_resnet_monodepth(self):
    model = gluoncv.model_zoo.get_model('monodepth2_resnet18_kitti_stereo_640x192',
                                        pretrained_base=False, ctx=self.ctx, pretrained=True)
    return model
  @staticmethod
  def load_image(filename):
    img = pil.open(filename).convert('RGB')
    return img

  def estimate_depth(self, filename, plot=True):
    img = self.load_image(filename)
    if plot:
      plt.figure()
      plt.imshow(img)
    output = self.predict(img)
    if plot:
      plt.figure()
      plt.imshow(output)
      
      
if __name__ == '__main__':
  filename = 'image.jpg'
  depth_model = DepthEstimator(width=640, height=192, ctx=0)
  depth_model.estimate_depth(filename, plot=True)
