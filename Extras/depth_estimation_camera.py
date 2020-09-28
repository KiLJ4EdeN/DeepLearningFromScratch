import PIL.Image as pil
import numpy as np
import mxnet as mx
from mxnet.gluon.data.vision import transforms
import gluoncv
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2


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

    def estimate_depth(self, filename, plot=True, is_image=False):
        if not is_image:
            img = self.load_image(filename)
        else:
            img = filename
        if plot:
            plt.figure()
            plt.imshow(img)
        output = self.predict(img)
        if plot:
            plt.figure()
            plt.imshow(output)
        return output


if __name__ == '__main__':
    depth_model = DepthEstimator(width=640, height=192, ctx=0)
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if ret:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            im_pil = pil.fromarray(rgb_frame)
            depth_image = depth_model.estimate_depth(im_pil, plot=False, is_image=True)
            depth_image = cv2.normalize(depth_image, None, alpha=0,
                                        beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            cv2.imshow('camera', frame)
            cv2.imshow('depth', depth_image)
            if cv2.waitKey(1) & 0XFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                break
        else:
            cap.release()
            cv2.destroyAllWindows()
