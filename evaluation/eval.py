import ast
import cv2
import copy
import numpy as np
import os
from datasets import VidSTGDataset, ShikraVideoDataset, HCSTVGDataset, NExTVideoDataset
from llava.constants import IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token
import torch
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings("ignore")

def load_model(pretrained, model_name, model_base, device, device_map):
    overwrite_config = {'tie_word_embeddings': False, 'use_cache': True, "vocab_size": 152064} if '7B' in model_name else None
    tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, model_base, model_name, load_4bit=False, torch_dtype="bfloat16", device_map=device_map, overwrite_config=overwrite_config)  # Add any other thing you want to pass in llava_model_args
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
            max_new_tokens=36000,
        )
    return cont

if __name__ == '__main__':
    split = "test"
    #dataset_path = f"/home/jfioresi/datasets/shikra_v/VidSTG-Dataset/annotations/{split}_annotations.json"
    dataset_path = f'/lustre/fs1/home/jfioresi/datasets/shikra_v/annotations/vidstg-it_{split}.json'

    pretrained = "/home/ttran/CAP/LLaVA-Video/work_dirs/llavavideo-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-vtimellm_50K_lora_a1"
    #pretrained = "/home/ttran/CAP/LLaVA-Video/work_dirs/llavanext-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-ov_to_video_elysium"
    #pretrained = "/home/jfioresi/vlm/LLaVA-Video/work_dirs/llavavideo-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-elysium_100K_lora_a1"
    model_name = os.path.basename(pretrained)
    #model_base = "lmms-lab/LLaVA-Video-7B-Qwen2"
    model_base = "/home/jfioresi/vlm/LLaVA-Video/work_dirs/llavavideo-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-elysium_100K_lora_a1"
    device = "cuda"
    device_map = "auto"
    max_frames_num = 8
    tokenizer, model, image_processor, max_length = load_model(pretrained, model_name, model_base, device, device_map)
    
    if 'VidSTG' in dataset_path:
        video_path = "/datasets/vidor/video"
        dataset = VidSTGDataset(dataset_path, video_path, split, image_processor, max_frames=max_frames_num)
        
    elif 'shikra_v' in dataset_path:
        video_path = "/datasets"
        dataset = ShikraVideoDataset(dataset_path, video_path, split, image_processor, max_frames=max_frames_num)
        
    elif 'HC-STVG' in dataset_path:
        video_path = "/groups/mshah/data/HC-STVG"
        dataset = HCSTVGDataset(dataset_path, video_path, split, image_processor, max_frames=max_frames_num)
        
    elif 'NExTVideo' in dataset_path:
        video_path = "/groups/mshah/data/NExTVideo"
        dataset = NExTVideoDataset(dataset_path, video_path, split, image_processor, max_frames=max_frames_num)
    else:
        raise Exception(f'Dataset specified has not yet been implemented.')

    dataloader = DataLoader(dataset, shuffle=False)
    
    for data in dataloader:
        # get question from dataset
        question_list = data['question']
        for index, question in enumerate(question_list):
            video = data['video']
            video = [video[0].squeeze()]

            # generate output from model
            conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models
            conv = copy.deepcopy(conv_templates[conv_template])
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
            prompt_question = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
            cont = generate(model, input_ids, video)
            text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()
            print(text_outputs)

            #output = text_outputs[text_outputs.index("{"):text_outputs.index("}") + 1]
            #frames = ast.literal_eval(output)

            exit()

            #testing with real bbox
            frames = dataset.bboxes
            #frames = dataset.bboxes[dataset.questions[index]['target_id']]
            
            #print(frames)
            
            # calculate IoU
            iou = dataset.calculate_iou(frames, index)

            print(iou)

