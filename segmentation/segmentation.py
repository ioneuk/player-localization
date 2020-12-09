import matplotlib.pyplot as plt

import detectron2
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer


class InstanceSegmentation:
    def __init__(self, score_thresh_test=0.5, device='cpu'):
        self.score_thresh_test = score_thresh_test
        self.device = device
        self.model_path = 'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yam'
        self.model, self.cfg = self.get_model()

    def get_model(self):
        cfg = get_cfg()
        cfg.MODEL.DEVICE = self.device
        cfg.merge_from_file(model_zoo.get_config_file(self.model_path))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.score_thresh_test
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(self.model_path)
        predictor = DefaultPredictor(cfg)
        return predictor, cfg

    def predict_sample(self, sample, plot=False, save_plot=False):
        output = self.model(sample)

        if plot:
            vis = Visualizer(
                sample[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1.2)
            out = vis.draw_instance_predictions(output['instances'].to('cpu'))
            plt.figure(figsize=(20, 18))
            plt.imshow(out.get_image()[:, :, ::-1])
            if save_plot:
                plt.savefig('sample.jpg')
            plt.show()

        return output


class PanopticSegmentation:
    def __init__(self, score_thresh_test=0.5, device='cpu'):
        self.score_thresh_test = score_thresh_test
        self.device = device
        self.model_path = 'COCO-PanopticSegmentation/panoptic_fpn_R_50_1x.yaml'
        self.model, self.cfg = self.get_model()

    def get_model(self):
        cfg = get_cfg()
        cfg.MODEL.DEVICE = self.device
        cfg.merge_from_file(model_zoo.get_config_file(self.model_path))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.score_thresh_test
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(self.model_path)
        predictor = DefaultPredictor(cfg)
        return predictor, cfg

    def predict_sample(self, sample, plot=False, save_plot=False):
        output = self.model(sample)

        if plot:
            vis = Visualizer(
                sample[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1.2)
            out = vis.draw_panoptic_seg_predictions(
                output["panoptic_seg"][0].to("cpu"), output["panoptic_seg"][1])
            plt.figure(figsize=(10, 8))
            plt.imshow(out.get_image()[:, :, ::-1])
            if save_plot:
                plt.savefig('sample.jpg')
            plt.show()

        return output


if __name__ == '__main__':
    import cv2
    test_img = cv2.imread('421.jpg')
    model = PanopticSegmentation()
    output = model.predict_sample(test_img, plot=True)
    print('Done!')
