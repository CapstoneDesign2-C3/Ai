import argparse
import os
import sys
import numpy as np
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from PIL import Image

# yolov7, strongsort utils import
from yolov7.models.experimental import attempt_load
from yolov7.utils.datasets import LoadImages, LoadStreams
from yolov7.utils.general import (check_img_size, non_max_suppression, scale_coords, check_requirements, cv2,
                                  check_imshow, xyxy2xywh, xywh2xyxy, clip_coords, increment_path, strip_optimizer, colorstr, check_file)
from yolov7.utils.torch_utils import select_device, time_synchronized
from yolov7.utils.plots import plot_one_box
from strong_sort.utils.parser import get_config
from strong_sort.strong_sort import StrongSORT

# 영상 포맷 정의
VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv'

# crop 저장 메서드
def save_one_box(xyxy, im, file=Path('im.jpg'), gain=1.02, pad=10, square=False, BGR=False, save=True):
    xyxy = torch.tensor(xyxy).view(-1, 4)
    b = xyxy2xywh(xyxy)
    if square:
        b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)
    b[:, 2:] = b[:, 2:] * gain + pad
    xyxy = xywh2xyxy(b).long()
    clip_coords(xyxy, im.shape)
    crop = im[int(xyxy[0, 1]):int(xyxy[0, 3]), int(xyxy[0, 0]):int(xyxy[0, 2]), ::(1 if BGR else -1)]
    
    if save:
        file.parent.mkdir(parents=True, exist_ok=True)
        f = str(Path(increment_path(file)).with_suffix('.jpg'))
        Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).save(f, quality=95, subsampling=0)
    
    return crop

class TrackerModule:
    def __init__(self, opt):
        self.opt = opt
        self.device = select_device(opt.device)
        self.half = opt.half
        self.strongsort_list = []
        self.outputs = []
        self.names = []
        self.model = None
        self.stride = None
        self.imgsz = None
        self.dataset = None
        self.nr_sources = None
        self.curr_frames = None
        self.prev_frames = None
        self.save_dir = None
        self.vid_path = None
        self.vid_writer = None
        self.txt_path = None
    
    def load_model(self):
        WEIGHTS = Path('weights')
        WEIGHTS.mkdir(parents=True, exist_ok=True)
        self.model = attempt_load(Path(self.opt.yolo_weights), map_location=self.device)
        self.names, = self.model.names,
        self.stride = self.model.stride.max().cpu().numpy()
        self.imgsz = check_img_size(self.opt.imgsz[0], s=self.stride)
    
    def save_crop_if_target(self, bboxes, imc, cls, id, p, frame_idx):
        if int(cls) in [0, 2]:
            video_name = p.stem
            save_folder = Path(f'{video_name}/{id}')
            save_folder.mkdir(parents=True, exist_ok=True)
            save_one_box(bboxes, imc, file=save_folder / f'frame_{frame_idx}.jpg', BGR=True)

    @torch.no_grad()
    def run(self):
        self.load_model()

        source = str(self.opt.source)
        is_file = Path(source).suffix[1:] in (VID_FORMATS)
        webcam = source.isnumeric()

        exp_name = Path(self.opt.yolo_weights).stem
        self.save_dir = increment_path(Path(self.opt.project) / self.opt.name, exist_ok=self.opt.exist_ok)
        (self.save_dir / 'tracks' if self.opt.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)

        if webcam:
            cudnn.benchmark = True
            self.dataset = LoadStreams(source, img_size=self.imgsz, stride=self.stride)
            self.nr_sources = len(self.dataset.sources)
        else:
            self.dataset = LoadImages(source, img_size=self.imgsz, stride=self.stride)
            self.nr_sources = 1

        self.curr_frames = [None] * self.nr_sources
        self.prev_frames = [None] * self.nr_sources
        self.vid_path = [None] * self.nr_sources
        self.vid_writer = [None] * self.nr_sources

        cfg = get_config()
        cfg.merge_from_file(self.opt.config_strongsort)
        for i in range(self.nr_sources):
            tracker = StrongSORT(
                self.opt.strong_sort_weights,
                self.device,
                self.half,
                max_dist=cfg.STRONGSORT.MAX_DIST,
                max_iou_distance=cfg.STRONGSORT.MAX_IOU_DISTANCE,
                max_age=cfg.STRONGSORT.MAX_AGE,
                n_init=cfg.STRONGSORT.N_INIT,
                nn_budget=cfg.STRONGSORT.NN_BUDGET,
                mc_lambda=cfg.STRONGSORT.MC_LAMBDA,
                ema_alpha=cfg.STRONGSORT.EMA_ALPHA,
            )
            tracker.model.warmup()
            self.strongsort_list.append(tracker)

        self.outputs = [None] * self.nr_sources

        for frame_idx, (path, im, im0s, vid_cap) in enumerate(self.dataset):
            im = torch.from_numpy(im).to(self.device)
            im = im.half() if self.half else im.float()
            im /= 255.0
            if len(im.shape) == 3:
                im = im[None]

            pred = self.model(im)
            pred = non_max_suppression(pred[0], self.opt.conf_thres, self.opt.iou_thres, self.opt.classes, self.opt.agnostic_nms)

            for i, det in enumerate(pred):
                if webcam:
                    p, im0, _ = path[i], im0s[i].copy(), self.dataset.count
                else:
                    p, im0, _ = path, im0s.copy(), getattr(self.dataset, 'frame', 0)

                self.curr_frames[i] = im0
                imc = im0.copy()

                if cfg.STRONGSORT.ECC:
                    self.strongsort_list[i].tracker.camera_update(self.prev_frames[i], self.curr_frames[i])

                if det is not None and len(det):
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                    xywhs = xyxy2xywh(det[:, 0:4])
                    confs = det[:, 4]
                    clss = det[:, 5]

                    self.outputs[i] = self.strongsort_list[i].update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)

                    if len(self.outputs[i]) > 0:
                        for j, (output, conf) in enumerate(zip(self.outputs[i], confs)):
                            bboxes = output[0:4]
                            id = output[4]
                            cls = output[5]

                            if self.opt.save_crop:
                                self.save_crop_if_target(bboxes, imc, cls, id, p, frame_idx)
                else:
                    self.strongsort_list[i].increment_ages()

                self.prev_frames[i] = self.curr_frames[i]

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-weights', nargs='+', type=str, default='weights/yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--strong-sort-weights', type=str, default='weights/osnet_x0_25_msmt17.pt')
    parser.add_argument('--config-strongsort', type=str, default='strong_sort/configs/strong_sort.yaml')
    parser.add_argument('--source', type=str, default='0', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/track', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--hide-class', default=False, action='store_true', help='hide IDs')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1
    return opt

def main(opt):
    ROOT = Path(__file__).resolve().parents[0]
    check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    tracker = TrackerModule(opt)
    tracker.run()

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
