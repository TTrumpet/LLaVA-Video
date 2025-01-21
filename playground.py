# pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git
import ast
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from PIL import Image
import requests
import copy
import cv2
import torch
import sys
import warnings
from decord import VideoReader, cpu
import numpy as np
import json
from visualize import visualize_output, unnormalize_bbox
warnings.filterwarnings("ignore")

def calculate_iou(boxes1: np.array, boxes2: np.array) -> float:
  """Calculate Intersection over Union (IoU) for two sets of bounding boxes.

  Args:
    boxes1 (np.array): Bounding boxes in the format [y2, x2, y1, x1],
      shape (n, 4)
    boxes2 (np.array): Bounding boxes in the format [y2, x2, y1, x1],
      shape (n, 4)

  Returns:
    iou (float): Intersection over Union (IoU) float value
  """
  x1_1, y1_1, x2_1, y2_1 = np.split(boxes1, 4, axis=1)
  x1_2, y1_2, x2_2, y2_2 = np.split(boxes2, 4, axis=1)

  # Find intersection coordinates
  y1_inter = np.maximum(y1_1, y1_2)
  x1_inter = np.maximum(x1_1, x1_2)
  y2_inter = np.minimum(y2_1, y2_2)
  x2_inter = np.minimum(x2_1, x2_2)

  # Calculate area of intersection
  h_inter = np.maximum(0, y2_inter - y1_inter)
  w_inter = np.maximum(0, x2_inter - x1_inter)
  area_inter = h_inter * w_inter

  # Calculate area of union
  area_boxes1 = (y2_1 - y1_1) * (x2_1 - x1_1)
  area_boxes2 = (y2_2 - y1_2) * (x2_2 - x1_2)
  union = area_boxes1 + area_boxes2 - area_inter

  return area_inter / union

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
        sample_fps = max_frames_num
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frame_time = [i/vr.get_avg_fps() for i in frame_idx]
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    # import pdb;pdb.set_trace()
    return spare_frames,frame_time,video_time

def load_model(pretrained, model_name, device, device_map):
    tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, load_4bit=False, torch_dtype="bfloat16", device_map=device_map)  # Add any other thing you want to pass in llava_model_args
    model.eval()
    # model = model.to(torch.bfloat16)
    return tokenizer, model, image_processor, max_length

def generate(model, input_ids, video):
    # with torch.inference_mode():
    cont = model.generate(
            input_ids,
            images=video,
            modalities= ["video"],
            do_sample=False,
            temperature=0,
            #max_new_tokens=4096,
            max_new_tokens=36000,
            )
    return cont

def main(split, pretrained):

    # initialize model
    #pretrained = "/lustre/fs1/home/ttran/CAP/LLaVA-Video/work_dirs/llavanext-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-ov_to_video_elysium/checkpoint-500/"
    model_name = "llavanext-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-ov_to_video_vidshikra"
    model_name_short = pretrained[pretrained.index("video") + 6 :]
    device = "cuda"
    device_map = "auto"
    max_frames_num = 64

    tokenizer, model, image_processor, max_length = load_model(pretrained, model_name, device, device_map)

    # read v_shikra test set json
    data = None
    with open(f'/lustre/fs1/home/jfioresi/datasets/shikra_v/annotations/vidstg-it_{split}.json') as json_data:
        data = json.load(json_data)
    if data == None:
        exit(1)

    # input and save set number of videos from v_shikra test set
    num_videos = 5
    outputs = {}

    for i in range(num_videos):
        print(i)

        # testing 1 video
        video_path = '/datasets/' + data[i]['video']
        video, frame_time, video_time = load_video(video_path, max_frames_num, 1, force_sample=True)
        video = image_processor.preprocess(video, return_tensors="pt")["pixel_values"].cuda().bfloat16()
        video = [video]
        
        #time_instruction = f"The video lasts for {video_time:.2f} seconds, and {len(video[0])} frames are uniformly sampled from it. These frames are located at {frame_time}. Please answer the following questions related to this video."
        #question = time_instruction + '\n' +  question
        question = data[i]['conversations'][0]['value'].replace('<video>', DEFAULT_IMAGE_TOKEN)
        
        conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models
        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
        cont = generate(model, input_ids, video)
        text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()

        output = text_outputs[text_outputs.index("{"):text_outputs.index("}") + 1]
        frames = ast.literal_eval(output)

        # visualize outputs
        # TODO: conda env does not work
        #gen_bbox = visualize_outputs(video_path, text_outputs, model_name=model_name_short)

        # calculate IoU
        cap = cv2.VideoCapture(video_path)
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ret, frame = cap.read()
        height, width, _  = frame.shape

        actual_bbox = data[i]['meta']['bboxes']

        num_gen_frames = len(frames)

        tmp_bbox = [] # real bboxes
        tmp_frames = [] # generated 100 bboxes

        for key, value in actual_bbox.items():

            # get closest frame in gen_bbox
            curr_frame = int((int(key) / num_frames) * num_gen_frames)
            if curr_frame >= num_gen_frames:
                curr_frame = num_gen_frames - 1

            actual = unnormalize_bbox(value, width, height)
            try:
                gen = unnormalize_bbox(frames[curr_frame], width, height)
            except:
                if curr_frame == 0:
                    gen = unnormalize_bbox(frames[curr_frame + 1], width, height)
            
            # reverse to get into IoU format 
            # [x1, y1, x2, y2] -> [y2, x2, y1, x1]
            #actual.reverse()
            #gen.reverse()

            tmp_bbox.append(np.array(actual))
            tmp_frames.append(np.array(gen))

        actual_bbox = np.array(tmp_bbox)
        gen_bbox = np.array(tmp_frames)

        iou = calculate_iou(gen_bbox, actual_bbox)
        #print(iou)
        print(np.average(iou))

        # save output to json file
        outputs[i] = {'video_path': video_path, 'text_outputs': text_outputs, 'iou_score': np.average(iou)}

    with open(f"{num_videos}_{model_name_short}_{split}_outputs.json", 'w') as f:
        json.dump(outputs, f)
    
if __name__ == '__main__':
    pretrained = "/lustre/fs1/home/ttran/CAP/LLaVA-Video/work_dirs/llavanext-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-ov_to_video_vidshikra_100"
    split = 'test'
    main(split, pretrained)
