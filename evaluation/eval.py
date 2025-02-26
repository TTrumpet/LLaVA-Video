import ast
import cv2
import copy
import json
import numpy as np
import os
from datasets import VidSTGDataset, ShikraVideoDataset, HCSTVGDataset, NExTVideoDataset, final_iou
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token
import torch
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings("ignore")

def load_model(pretrained, model_name, device, device_map=None, model_base=None):
    overwrite_config = {'tie_word_embeddings': False, 'use_cache': True, "vocab_size": 152064} if '7B' in model_name else None
    tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, model_base=model_base, model_name=model_name, load_4bit=False, torch_dtype="bfloat16", device_map=device_map, overwrite_config=overwrite_config)  # Add any other thing you want to pass in llava_model_args
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
    device = "cuda"
    device_map = "auto"
    max_frames_num = 100
    test_qa = False
    
    #dataset_path = f"/home/jfioresi/datasets/shikra_v/VidSTG-Dataset/annotations/{split}_annotations.json"
    #dataset_path = f'/lustre/fs1/home/jfioresi/datasets/shikra_v/annotations/vidstg-it_{split}.json'
    dataset_path = f"/home/jfioresi/datasets/NExT-GQA/gsub_{split}.json"
    # dataset_path = "/groups/mshah/data/HC-STVG/anno_v2/val_v2.json"

    pretrained = "/home/jfioresi/vlm/LLaVA-Video/work_dirs/llavavideo-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-vtimellm_50K_lora_a1"
    #pretrained = "/home/ttran/CAP/LLaVA-Video/work_dirs/llavanext-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-ov_to_video_elysium"
    # pretrained = "/home/jfioresi/vlm/LLaVA-Video/work_dirs/llavavideo-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-elysium_100K_lora_a1"
    # pretrained = "/home/jfioresi/vlm/LLaVA-Video/work_dirs/llavavideo-Qwen2-7B_elysium_bbox_merged"
    
    model_name = os.path.basename(pretrained)
    
    #model_base = "lmms-lab/LLaVA-Video-7B-Qwen2"
    #model_base = "/home/jfioresi/vlm/LLaVA-Video/work_dirs/llavavideo-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-elysium_100K_lora_a1"
    model_base = "/home/jfioresi/vlm/LLaVA-Video/work_dirs/llavavideo-Qwen2-7B_elysium_bbox_merged_a2"

    tokenizer, model, image_processor, max_length = load_model(pretrained, model_name, device, device_map=device_map, model_base=model_base)
    
    if 'VidSTG' in dataset_path:
        video_path = "/datasets/vidor/video"
        vidor_anno_path = "/lustre/fs1/home/ttran/CAP/datasets/VidSTG-Dataset/validation/"
        dataset = VidSTGDataset(dataset_path, video_path, split, image_processor, max_frames=max_frames_num, vidor_anno_path=vidor_anno_path)
        
    elif 'shikra_v' in dataset_path:
        video_path = "/datasets"
        dataset = ShikraVideoDataset(dataset_path, video_path, split, image_processor, max_frames=max_frames_num)
        
    elif 'HC-STVG' in dataset_path:
        video_path = "/groups/mshah/data/HC-STVG/VIdeo"
        dataset = HCSTVGDataset(dataset_path, video_path, split, image_processor, max_frames=max_frames_num)
        
    elif 'NExT-GQA' in dataset_path:
        video_path = "/groups/mshah/data/NExTVideo"
        # map_id_to_path_file = "/home/ttran/CAP/datasets/NExT-GQA/map_vid_vidorID.json"
        # test_qa_file = "/home/ttran/CAP/datasets/NExT-GQA/test.csv"
        map_id_to_path_file = "/home/jfioresi/datasets/NExT-GQA/map_vid_vidorID.json"
        test_qa_file = "/home/jfioresi/datasets/NExT-GQA/test.csv"

        dataset = NExTVideoDataset(dataset_path, video_path, split, image_processor, max_frames=max_frames_num, map_id_to_path_file=map_id_to_path_file, test_qa_file=test_qa_file, test_qa=test_qa)
    else:
        raise Exception(f'Dataset specified has not yet been implemented.')

    dataloader = DataLoader(dataset, shuffle=False)

    results = {}
    test_qa = False
    output_file = 'evaluation/vtimellm_results.json'
    
    # TODO: Add try except in case the generation fails/not in accepted format
    for data in dataloader:
        # get question from dataset
        question_list = data['question']
        if data['video_path'][0] not in results:
            results[data['video_path'][0]] = []
        for index, question in enumerate(question_list):
            try:
                video = data['video']
                video_time = data['video_time'][0].item()
                frame_time = data['frame_time']
                video = [video[0].squeeze()]
                question += "? What frames does this occur?"

                # generate output from model
                conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models
                conv = copy.deepcopy(conv_templates[conv_template])
                conv.append_message(conv.roles[0], question)
                conv.append_message(conv.roles[1], None)
                prompt_question = conv.get_prompt()
                time_instruction = f"The video lasts for {video_time:.2f} seconds, and {len(video[0])} frames are sampled from it. These frames are located at {frame_time}. Please answer the following questions related to this video."
                prompt_question = f'{DEFAULT_IMAGE_TOKEN}\n{time_instruction}\n{prompt_question.replace("<video>", "")}'
                print(prompt_question)
                
                input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
                cont = generate(model, input_ids, video)
                text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()
                print(text_outputs)
                pred_start, pred_end = text_outputs.split(' to ')
                pred_start = pred_start.replace('From ', '')
                pred_end = pred_end.replace('.', '')
                gt_start, gt_end = data['duration'][index][0].item(), data['duration'][index][1].item()
                print('Question:', question, 'Output:', text_outputs)
                print('pred start:', pred_start, 'pred_end:', pred_end)
                print('gt_start:', gt_start, 'gt_end:', gt_end)

                results[data['video_path'][0]].append({
                    'pred_start': pred_start,
                    'pred_end': pred_end,
                    'gt_start': gt_start,
                    'gt_end': gt_end,
                })
                with open(output_file, 'w') as f:
                    json.dump(results, f)

            except Exception as e:
                # print(e)
                pass

            #output = text_outputs[text_outputs.index("{"):text_outputs.index("}") + 1]
            #frames = ast.literal_eval(output)

            #testing with real bbox
            # frames = dataset.bboxes
            #frames = dataset.bboxes[dataset.questions[index]['target_id']]
            
            #print(frames)
            
            # calculate IoU
            # iou = dataset.calculate_iou(frames, index)
            # print(iou)

            # evaluate qa response
            if test_qa:
                dataset.evaluate_qa(text_outputs)
        # break


    # output final iou
    # print(final_iou(dataset.iou))
    
    # output final qa accuracy
    if test_qa:
       print(dataset.qa_accuracy / len(dataset))

