from csv import DictReader
import cv2
import json
import numpy as np
import re
import os

from llava.constants import DEFAULT_IMAGE_TOKEN

import torch
import torchvision
from torchmetrics.detection import IntersectionOverUnion
from torch.utils.data import Dataset
import zipfile

from decord import VideoReader, cpu

def negative_check(num_list):
    n = [num for num in num_list if num < 0]
    if len(n) == 0:
        return True
    return False

# Convert nested bbox and labels in dictionary 
# to correct format for IoU calculation.
def nested_dict_to_list(given_dict):
    
    boxes = []
    for bbox in given_dict.values():
        new_bbox = []
        for num in bbox.values():
            new_bbox.append(num)
        #assert(negative_check(new_bbox))
        boxes.append(new_bbox)
            
    labels = []
    for label in given_dict.keys():
        labels.append(int(label))
    #assert(negative_check(labels))
    
    return boxes, labels

# Convert bbox and labels in dictionary 
# to correct format for IoU calculation.    
def dict_to_list(given_dict):
    
    boxes = []
    for bbox in given_dict.values():
        #assert(negative_check(bbox))
        boxes.append(bbox)
            
    labels = []
    for label in given_dict.keys():
        labels.append(int(label))
    #assert(negative_check(labels))
    
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


def load_video(video_path, max_frames_num,fps=1, force_sample=False):
    if max_frames_num == 0:
        return np.zeros((1, 336, 336, 3))
    vr = VideoReader(video_path, ctx=cpu(0),num_threads=1)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    fps = round(vr.get_avg_fps()/fps)
    frame_idx = [i for i in range(0, len(vr), fps)]
    frame_time = [i/fps for i in frame_idx]
    if len(frame_idx) > max_frames_num: # or force_sample:
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
    
def load_video_zip(video_path, max_frames_num, fps=1, VIDEO_FPS=3, force_sample=False):
    assert fps <= VIDEO_FPS
    # parse out zip file path from rest of path
    path = video_path.split('.zip')
    zip_path = path[0] + '.zip'
    folder = path[1].replace('/', '')

    with zipfile.ZipFile(zip_path, 'r') as z:
        frame_list = [f for f in z.namelist() if folder in f and '.jpg' in f]
        # FPS calc
        total_frame_num = len(frame_list)
        video_time = total_frame_num / VIDEO_FPS
        frame_idx = [i for i in range(0, total_frame_num, VIDEO_FPS)]
        frame_time = [i/fps for i in frame_idx]
        duration = len(frame_idx)
        start_idx = 0

        if max_frames_num > 0:
            if len(frame_idx) > max_frames_num:
                # Choose random index, sample next frames_upbound frames
                start_idx = np.random.randint(0, len(frame_idx) - max_frames_num)
                frame_idx = frame_idx[start_idx:start_idx + max_frames_num]
                frame_time = frame_time[start_idx:start_idx + max_frames_num]

        full_clip = []
        for idx in frame_idx:
            frame = torchvision.io.decode_image(torch.frombuffer(bytearray(z.read(frame_list[idx])), dtype=torch.uint8))
            full_clip.append(frame)
        full_clip = torch.stack(full_clip, dim=0)

        frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
        num_frames_to_sample = num_frames = len(frame_idx)

    return full_clip, frame_time, video_time


def final_iou(iou_dict):
    iou = 0
    for t in iou_dict.values():
        iou += t
    iou = iou / len(iou_dict)
    return iou


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
        
        # save values to use in iou calculation
        self.bboxes = data_item['meta']['bboxes']
        self.vid = data_item['id']
        
        return {
            'question': [DEFAULT_IMAGE_TOKEN + "\n" + data_item['conversations'][0]['value']],
            'video': video,
            'bboxes': data_item['meta']['bboxes']
            }
        
    def calculate_iou(self, gen_frames, question_index=None):
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
    def __init__(self, dataset_path, video_path, split, image_processor, max_frames=8, video_processor=None, vidor_anno_path=None):
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
        # TODO: move this to load_dataset
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
            
            db_list = []
            for vid, val in db_file.items():
                # get path to video
                for path, folders, files in os.walk(self.video_path):
                    for filename in files:
                        if filename == vid:
                            video_path = path
                val['video_path'] = os.path.join(video_path, vid)
                val['vid'] = vid.replace('.mp4', '')
                db_list.append(val)
            
            return db_list
            
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        data_item = self.db[idx]
        video_path = data_item['video_path']
        
        # create video tensor
        video, frame_time, video_time = load_video(video_path, self.max_frames, fps=1, force_sample=True)
        video = self.image_processor.preprocess(video, return_tensors="pt")["pixel_values"].cuda().bfloat16()
        video = [video]
        
        bbox_dict = {}
        # convert to {frame_id: [bboxes], ...} format
        # convert [x, y, w, h] -> [x1, y1, x2, y2] format
        counter = 0

        for frame in range(data_item['st_frame'], data_item['ed_frame']):
            tmp_bbox = data_item['bbox'][counter]
            tmp_bbox = [tmp_bbox[0], tmp_bbox[1], tmp_bbox[0] + tmp_bbox[2], tmp_bbox[1] + tmp_bbox[3]]
            bbox_dict[frame] = tmp_bbox
            counter += 1
        
        # save values to use in iou calculation
        self.bboxes = bbox_dict
        self.vid = data_item['vid']
        
        return {
                "video": video,
                "question": data_item['English'],
                "bboxes": bbox_dict, 
            }
        
    def calculate_iou(self, gen_frames, question_index=None):
        #boxes, labels = dict_to_list(gen_frames)
        boxes, labels = dict_to_list(self.bboxes)
        preds = [
            {
                "boxes": torch.tensor(boxes),
                "labels": torch.tensor(labels),
            }
        ]
        
        boxes, labels = dict_to_list(self.bboxes)
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
        
        
class NExTVideoDataset(Dataset):
    def __init__(self, dataset_path, video_path, split, image_processor, max_frames=8, video_processor=None, map_id_to_path_file=None, test_qa_file=None, test_qa=False):
        self.dataset_path = dataset_path
        self.video_path = video_path
        self.id_map_path = map_id_to_path_file
        self.test_qa_path = test_qa_file
        self.test_qa = test_qa
        self.db = self.load_dataset()
        self.split = split
        self.max_frames = max_frames
        self.image_processor = image_processor
        self.video_processor = video_processor
        self.vid = None
        self.gt_ground = None
        self.iou = {}
        self.qa_correct = None
        self.qa_accuracy = 0
    
    def load_dataset(self):
        if not os.path.isfile(self.dataset_path):
            raise FileNotFoundError(f'No such file: {self.dataset_path}')
        if not os.path.isfile(self.id_map_path):
            raise FileNotFoundError(f'No such file: {self.id_map_path}')
        if not os.path.isfile(self.test_qa_path):
            raise FileNotFoundError(f'No such file: {self.test_qa_path}')
        with open(self.dataset_path, 'r') as f:
            gsub_file = json.load(f)
            self.len = len(gsub_file)
        with open(self.id_map_path, 'r') as f:
            id_map_file = json.load(f)
        with open(self.test_qa_path, 'r') as f:
            dict_reader = DictReader(f)
            qa_file = list(dict_reader)
        
        db_list = []
        for video in qa_file:
            # map video id (qa csv) to video path (id map)
            video_id = video['video_id']
            video['video_path'] = os.path.join(self.video_path, id_map_file[video_id]+".mp4")
            # map video id (qa csv) to temporal boundings
            tmp_gsub = gsub_file[video_id]
            # map qid (qa csv) to time interval (gsub json)
            qid = video['qid']
            video['gsub'] = {
                    "duration": tmp_gsub["duration"],
                    "location": tmp_gsub["location"][str(qid)]
                }
            video['answers'] = [video['a0'], video['a1'], video['a2'], video['a3'], video['a4']]
            video['answer_idx'] = video['answers'].index(video['answer'])
            db_list.append(video)

        return db_list
            
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        data_item = self.db[idx]
        video_path = data_item['video_path']
        
        # create video tensor
        video, frame_time, video_time = load_video(video_path, self.max_frames, fps=1, force_sample=True)
        # print(len(video), frame_time, video_time)
        # print(data_item['gsub']['location'])
        video = self.image_processor.preprocess(video, return_tensors="pt")["pixel_values"].cuda().bfloat16()
        video = [video]

        # Save variables to be used in iou calculation
        self.gt_ground = data_item['gsub']['location']
        self.vid = data_item['qid']
        
        # QA testing
        answer = data_item['answers'][int(data_item['answer_idx'])]
        self.qa_correct = {data_item['answer_idx']: answer}
        
        return {
                "video": video,
                "question": data_item['question'],
                "duration": data_item['gsub']['location'], 
                "video_path": data_item['video_path'],
                "frame_time": frame_time,
                "video_time": video_time
            }
        
    def get_tIoU(self, loc, span):

        if span[0] == span[-1]:
            if loc[0] <= span[0] and span[0] <= loc[1]:
                return 0, 1
            else:
                return 0, 0

        span_u =  (min(loc[0], span[0]), max(loc[-1], span[-1]))
        span_i = (max(loc[0], span[0]), min(loc[-1], span[-1]))
        dis_i = (span_i[1] - span_i[0])
        if span_u[1] > span_u[0]:
            IoU = dis_i / (span_u[1] - span_u[0])
        else:
            IoU = 0.0
        if span[-1] > span[0]:
            IoP = dis_i / (span[-1] - span[0])
        else:
            IoP = 0.0

        return IoU, IoP

    # Assumes input gt_ground: [[x,y]], pred_ground: [x,y]
    # technically tIoU
    def eval_ground(self, gt_ground, pred_ground):      
        span = pred_ground
        tIoU, tIoP = self.get_tIoU(gt_ground[0], span)
        self.iou[self.vid] = IoU
        return tIoU

    def calculate_iou(self, gen_ground, question_index=None):
        return self.eval_ground(self.gt_ground, gen_ground)

    def evaluate_qa(self, gen_response):
        print(self.qa_correct)
        for idx, val in self.qa_correct.items():
            if idx in gen_response or 'a'+str(idx) in gen_response or val in gen_response:
                self.qa_accuracy += 1
                return True
            elif re.sub(r'[^\w\s]', '', val) in gen_response:
                self.qa_accuracy += 1
                return True
            return False
        
        
class TVQAPlusDataset(Dataset):
    def __init__(self, dataset_path, video_path, split, image_processor, max_frames=8, video_processor=None, test_qa=False):
        self.dataset_path = dataset_path
        self.video_path = video_path
        self.test_qa = test_qa
        self.db = self.load_dataset()
        self.split = split
        self.max_frames = max_frames
        self.image_processor = image_processor
        self.video_processor = video_processor
        self.vid = None
        self.gt_ground = None
        self.iou = {}
        self.qa_correct = None
        self.qa_accuracy = 0
    
    def load_dataset(self):
        if not os.path.isfile(self.dataset_path):
            raise FileNotFoundError(f'No such file: {self.dataset_path}')
        with open(self.dataset_path, 'r') as f:
            db_file = json.load(f)
            self.len = len(db_file)
        
        db_list = []
        for video in db_file:
            video['video_path'] = os.path.join(self.video_path, video['vid_name'])
            db_list.append(video)

        return db_list
            
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        data_item = self.db[idx]
        video_path = data_item['video_path']
        
        # create video tensor
        video, frame_time, video_time = load_video_zip(video_path, self.max_frames, fps=1, force_sample=True)
        video = self.image_processor.preprocess(video, return_tensors="pt")["pixel_values"].cuda().bfloat16()
        video = [video]
        
        # get bboxes
        total_normalize_frame_to_bbox = {}
        
        # get objects in answer
        answers = [data_item['a0'], data_item['a1'], data_item['a2'], data_item['a3'], data_item['a4']]
        answer = answers[int(data_item['answer_idx'])]

        for l, val in data_item['bbox'].items():
            if len(val) == 0:
                continue

            normalize_frame_to_bbox = {}
            for bbox in val:
                if bbox['label'] in answer:
                    xmin = bbox['left']
                    xmax = bbox['left'] + bbox['width']
                    ymin = bbox['top']
                    ymax = bbox['top'] + bbox['height']
                    normalize_frame_to_bbox[l] = [xmin, ymin, xmax, ymax]
                    if bbox['label'] in total_normalize_frame_to_bbox:
                        total_normalize_frame_to_bbox[bbox['label']][l] = [xmin, ymin, xmax, ymax]
                    else:
                        total_normalize_frame_to_bbox[bbox['label']] = normalize_frame_to_bbox

        # if empty, just put any bboxes
        if len(total_normalize_frame_to_bbox) == 0:
            for l, val in data_item['bbox'].items():
                if len(val) == 0:
                    continue

                normalize_frame_to_bbox = {}
                for bbox in val:
                    if bbox['label'] in answer:
                        xmin = bbox['left']
                        xmax = bbox['left'] + bbox['width']
                        ymin = bbox['top']
                        ymax = bbox['top'] + bbox['height']
                        normalize_frame_to_bbox[l] = [xmin, ymin, xmax, ymax]
                        if bbox['label'] in total_normalize_frame_to_bbox:
                            total_normalize_frame_to_bbox[bbox['label']][l] = [xmin, ymin, xmax, ymax]
                        else:
                            total_normalize_frame_to_bbox[bbox['label']] = normalize_frame_to_bbox


        # Save variables to be used in iou calculation
        self.gt_ground = data_item['ts']
        self.vid = data_item['qid']
        self.bboxes = total_normalize_frame_to_bbox
        
        # QA variables
        self.qa_correct = {data_item['answer_idx']: answer}
        if self.test_qa:
            data_item['q'] = data_item['q'] + '\n1. ' + data_item['a0'] + '\n2. ' + data_item['a1'] + '\n3. ' + data_item['a2'] + '\n4. ' + data_item['a3']  + '\n5. ' + data_item['a4']
        
        return {
                "video": video,
                "question": data_item['q'],
                "duration": [data_item['ts']], 
                "video_path": data_item['video_path'],
                "frame_time": frame_time,
                "video_time": video_time
            }
        
    def get_tIoU(self, loc, span):

        if span[0] == span[-1]:
            if loc[0] <= span[0] and span[0] <= loc[1]:
                return 0, 1
            else:
                return 0, 0

        span_u =  (min(loc[0], span[0]), max(loc[-1], span[-1]))
        span_i = (max(loc[0], span[0]), min(loc[-1], span[-1]))
        dis_i = (span_i[1] - span_i[0])
        if span_u[1] > span_u[0]:
            IoU = dis_i / (span_u[1] - span_u[0])
        else:
            IoU = 0.0
        if span[-1] > span[0]:
            IoP = dis_i / (span[-1] - span[0])
        else:
            IoP = 0.0

        return IoU, IoP

    # Assumes input gt_ground: [x,y], pred_ground: [x,y]
    # technically tIoU
    def eval_ground(self, gt_ground, pred_ground):      
        span = pred_ground
        tIoU, tIoP = self.get_tIoU(gt_ground, span)
        self.iou[self.vid] = IoU
        return tIoU

    def calculate_tiou(self, gen_ground, question_index=None):
        return self.eval_ground(self.gt_ground, gen_ground)
        
    def calculate_iou(self, gen_frames, question_index=None):
        #boxes, labels = dict_to_list(gen_frames)
        boxes, labels = dict_to_list(self.bboxes)
        preds = [
            {
                "boxes": torch.tensor(boxes),
                "labels": torch.tensor(labels),
            }
        ]
        
        boxes, labels = dict_to_list(self.bboxes)
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

    # TODO: edit llm prompt to only allow responses from 1 to 5
    def evaluate_qa(self, gen_response):
        print(self.qa_correct)
        for idx, val in self.qa_correct.items():
            if idx in gen_response or 'a'+str(idx) in gen_response or val in gen_response:
                self.qa_accuracy += 1
                return True
            elif re.sub(r'[^\w\s]', '', val) in gen_response:
                self.qa_accuracy += 1
                return True
            return False
