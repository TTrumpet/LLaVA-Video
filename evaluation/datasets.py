import cv2
import json
import numpy as np
import re
import os

from llava.constants import DEFAULT_IMAGE_TOKEN

import torch
from torchmetrics.detection import IntersectionOverUnion
from torch.utils.data import Dataset

from decord import VideoReader, cpu

# Convert nested bbox and labels in dictionary 
# to correct format for IoU calculation.
def nested_dict_to_list(given_dict):
    
    boxes = []
    for bbox in given_dict.values():
        new_bbox = []
        for num in bbox.values():
            new_bbox.append(num)
        boxes.append(new_bbox)
            
    labels = []
    for label in given_dict.keys():
        labels.append(int(label))
    
    return boxes, labels

# Convert bbox and labels in dictionary 
# to correct format for IoU calculation.    
def dict_to_list(given_dict):
    
    boxes = []
    for bbox in given_dict.values():
        boxes.append(bbox)
            
    labels = []
    for label in given_dict.keys():
        labels.append(int(label))
    
    return boxes, labels

# Normalize the bounding box to 0-100 scale.
def normalize_bbox(bbox, width, height):
    """
    Normalize the bbox
    """
    xmin, ymin, xmax, ymax = bbox
    xmin = int(round(xmin / width, 2) * 100)
    ymin = int(round(ymin / height, 2) * 100)
    xmax = int(round(xmax / width, 2) * 100)
    ymax = int(round(ymax / height, 2) * 100)

    return [xmin, ymin, xmax, ymax]


# Return bounding box to original scale.
def unnormalize_bbox(bbox, width, height):
    """
    Unnormalize the bbox
    """
    xmin, ymin, xmax, ymax = bbox
    xmin = int(round(xmin / 100 * width, 2))
    ymin = int(round(ymin / 100 * height, 2))
    xmax = int(round(xmax / 100 * width, 2))
    ymax = int(round(ymax / 100 * height, 2))

    return [xmin, ymin, xmax, ymax]


def load_video(video_path, max_frames_num,fps=1,force_sample=False):
    if max_frames_num == 0:
        return np.zeros((1, 336, 336, 3))
    vr = VideoReader(video_path, ctx=cpu(0),num_threads=1)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    fps = round(vr.get_avg_fps()/fps)
    frame_idx = [i for i in range(0, len(vr), fps)]
    frame_time = [i/fps for i in frame_idx]
    if len(frame_idx) > max_frames_num or force_sample:
        # sample_fps = max_frames_num
        # uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
        # frame_idx = uniform_sampled_frames.tolist()
        # frame_time = [i/vr.get_avg_fps() for i in frame_idx]
        start_idx = np.random.randint(0, len(frame_idx) - max_frames_num)
        frame_idx = frame_idx[start_idx:start_idx + max_frames_num]
        frame_time = frame_time[start_idx:start_idx + max_frames_num]

    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    # import pdb;pdb.set_trace()
    return spare_frames,frame_time,video_time


class ShikraVideoDataset(Dataset):
    def __init__(self, dataset_path, video_path, split, image_processor, max_frames=8, video_processor=None):
        self.dataset_path = dataset_path
        self.video_path = video_path
        self.db = self.load_dataset()
        self.split = split
        self.max_frames = max_frames
        self.image_processor = image_processor
        self.video_processor = video_processor
        self.iou = {}
    
    def load_dataset(self):
        if not os.path.isfile(self.dataset_path):
            raise FileNotFoundError(f'No such file: {self.dataset_path}')
        with open(self.dataset_path, 'r') as f:
            db_file = json.load(f)
            self.len = len(db_file)
            return db_file
            
    def convert(self, duration, x, num_frames=100):
        x = x / duration * num_frames
        x = str(min(round(x), num_frames - 1))
        # TODO: need to adjust if we ever go above 100 frames.
        if len(x) == 1:
            x = "0" + x
        return x
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        data_item = self.db[idx]
        
        video_path = os.path.join(self.video_path, data_item['video'])
        
        # create video tensor
        video, frame_time, video_time = load_video(video_path, self.max_frames, fps=1, force_sample=True)
        video = self.image_processor.preprocess(video, return_tensors="pt")["pixel_values"].cuda().bfloat16()
        video = [video]
        
        # Replace the tokens with normalized frame ids.
        replace_set = []
        for k, v in data_item['meta']['token'].items():
            replace_set.append((k, self.convert(data_item['meta']['duration'], v, len(data_item['meta']['bboxes']))))
        for x1, x2 in replace_set:
            for sentence in data_item['conversations']:
                sentence["value"] = sentence["value"].replace(x1, x2)
        
        # Replace bounding boxes.
        all_bbox_fid = list(data_item['meta']['bboxes'].keys())
        all_norm_fid = [x[1] for x in replace_set]
        all_norm_fid = [str(x) for x in range(int(all_norm_fid[0]), int(all_norm_fid[1]) + 1)]
        selected_bbox_fid = [x for x in range(int(all_bbox_fid[0]), int(all_bbox_fid[-1]) + 1, len(all_bbox_fid) // (len(all_norm_fid) - 1))]
        normalize_frame_to_bbox = {}
        for fid, norm_fid in zip(selected_bbox_fid, all_norm_fid):
            normalize_frame_to_bbox[int(norm_fid)] = data_item['meta']['bboxes'][str(fid)]
        bbox_string = re.sub(r'\s+', '', str(normalize_frame_to_bbox))
        for sentence in data_item['conversations']:
            sentence["value"] = sentence["value"].replace('<bboxes>', bbox_string)
        
        print(data_item['conversations'])
        
        # save values to use in iou calculation
        self.bboxes = data_item['meta']['bboxes']
        self.vid = data_item['id']
        
        return {
            'question': [DEFAULT_IMAGE_TOKEN + "\n" + data_item['conversations'][0]['value']],
            'video': video,
            'bboxes': data_item['meta']['bboxes']
            }
        
    def calculate_iou(self, gen_frames, question_index):
        assert isinstance(gen_frames, dict)
        boxes, labels = dict_to_list(gen_frames)  
              
        preds = [
            {
                "boxes": torch.tensor(boxes),
                "labels": torch.tensor(labels),
            }
        ]
        
        actual_bbox = self.bboxes   
        boxes, labels = dict_to_list(actual_bbox)
            
        target = [
            {
                "boxes": torch.tensor(boxes),
                "labels": torch.tensor(labels)
            }
        ]
            
        metric = IntersectionOverUnion()
        iou = metric(preds, target)
        self.iou[self.vid] = iou['iou']
            
        return iou
        

class VidSTGDataset(Dataset):
    def __init__(self, dataset_path, video_path, split, image_processor, max_frames=8, video_processor=None, vidor_anno_path="/lustre/fs1/home/ttran/CAP/datasets/VidSTG-Dataset/validation/"):
        self.dataset_path = dataset_path
        self.vidor_anno_path = vidor_anno_path
        self.video_path = video_path
        self.db = self.load_dataset()
        self.split = split
        self.max_frames = max_frames
        self.image_processor = image_processor
        self.video_processor = video_processor
        self.iou = {}

    def load_dataset(self):
        if not os.path.isfile(self.dataset_path):
            raise FileNotFoundError(f'No such file: {self.dataset_path}')
        with open(self.dataset_path, 'r') as f:
            db_file = json.load(f)
            self.len = len(db_file)
            if isinstance(db_file, dict):
                db_list = []
                for _, val in db_dict.items():
                    db_list.append(val)
                return db_list
            if isinstance(db_file, list):
                return db_file

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        data_item = self.db[idx]
        
        # Search VidOR annotations for video path
        vidor_filename = None
        for path, folders, files in os.walk(self.vidor_anno_path):
            for filename in files:
                cur_filename = filename.replace('.json', '')
                if cur_filename == data_item['vid']:
                    vidor_filename = os.path.join(path, filename)
        vidor_annotation_path = vidor_filename
        
        # read annotations to get video_path and trajectories
        vidor_data = None
        with open(vidor_annotation_path, 'r') as f:
            vidor_data = json.load(f)
        
        video_path = os.path.join(self.video_path, vidor_data['video_path'])
        trajectories = vidor_data['trajectories']

        # create video tensor
        video, frame_time, video_time = load_video(video_path, self.max_frames, fps=1, force_sample=True)
        video = self.image_processor.preprocess(video, return_tensors="pt")["pixel_values"].cuda().bfloat16()
        video = [video]
        
        # bounding boxes in format {id: {frame_idx : [bbox] ...} ...}
        bboxes = {}
        begin_fid = data_item["temporal_gt"]["begin_fid"]
        end_fid = data_item["temporal_gt"]["end_fid"]
        
        for q in data_item['questions']:
            target_id = q['target_id']
            
            frame_dict = {}
            frame_count = 0
            for trajectory in trajectories:
                if frame_count < begin_fid:
                    frame_count += 1
                    continue
                elif frame_count >= end_fid:
                    break
                for t in trajectory:
                    if t['tid'] == target_id:
                        frame_dict[frame_count] = t['bbox']
                frame_count += 1
            bboxes[target_id] = frame_dict
            
        # save values to use in iou calculation
        self.questions = data_item['questions']
        self.bboxes = bboxes
        self.vid = data_item['vid']
        
        # save question in correct format
        questions_list = []
        for question in self.questions:
            questions_list.append(DEFAULT_IMAGE_TOKEN + "\n" + question['description'])

        return {
                'video': video,
                #'height': data_item['height'],
                #'width': data_item['width'],
                #'num_frames': data_item['frame_count'],
                #'fps': data_item['fps'],
                #'input_segment': data_item['used_segment'],
                'question': questions_list,
                #'answer': data_item['captions'],
                #'temporal_gt': data_item['temporal_gt'],
                #'trajectories': trajectories,
                'bboxes': bboxes
                #'metadata': {
                #    'subjects': data_item['subject/objects'],
                #    'used_relation': data_item['used_relation']
                #    }
                }

    def calculate_iou(self, gen_frames, question_index):
        assert isinstance(gen_frames, dict)
        boxes, labels = nested_dict_to_list(gen_frames)       
        
        preds = [
            {
                "boxes": torch.tensor(boxes),
                "labels": torch.tensor(labels),
            }
        ]
        
        actual_bbox = self.bboxes[self.questions[question_index]['target_id']]    
        boxes, labels = nested_dict_to_list(actual_bbox)
            
        target = [
            {
                "boxes": torch.tensor(boxes),
                "labels": torch.tensor(labels)
            }
        ]
            
        metric = IntersectionOverUnion()
        iou = metric(preds, target)
        self.iou[self.vid] = iou['iou']
            
        return iou
    

class HCSTVGDataset(Dataset):
    def __init__(self, dataset_path, video_path, split, image_processor, max_frames=8, video_processor=None):
        self.dataset_path = dataset_path
        self.video_path = video_path
        self.db = self.load_dataset()
        self.split = split
        self.max_frames = max_frames
        self.image_processor = image_processor
        self.video_processor = video_processor
        self.iou = {}
    
    def load_dataset(self):
        if not os.path.isfile(self.dataset_path):
            raise FileNotFoundError(f'No such file: {self.dataset_path}')
        with open(self.dataset_path, 'r') as f:
            db_file = json.load(f)
            self.len = len(db_file)
            print(db_file)
            exit()
            
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        data_item = self.db[idx]
        
    def calculate_iou(self, gen_frames):
        pass
        
        
class NExTVideoDataset(Dataset):
    def __init__(self, dataset_path, video_path, split, image_processor, max_frames=8, video_processor=None):
        self.dataset_path = dataset_path
        self.video_path = video_path
        self.db = self.load_dataset()
        self.split = split
        self.max_frames = max_frames
        self.image_processor = image_processor
        self.video_processor = video_processor
        self.iou = {}
    
    def load_dataset(self):
        if not os.path.isfile(self.dataset_path):
            raise FileNotFoundError(f'No such file: {self.dataset_path}')
        with open(self.dataset_path, 'r') as f:
            db_file = json.load(f)
            self.len = len(db_file)
            print(db_file)
            exit()
            
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        data_item = self.db[idx]
        
    def calculate_iou(self, gen_frames):
        pass
