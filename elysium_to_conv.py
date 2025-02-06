import json
import random


new_file = "conv_ElysiumTrack-100K-Newton.json"
new_json = []

# First read all lines into memory
with open("/lustre/fs1/home/jfioresi/datasets/elysium_track/ElysiumTrack-1M-Newton.json", 'r') as f:
    all_lines = f.readlines()

# Randomly sample 100K lines
sampled_lines = random.sample(all_lines, 100000)

for line in sampled_lines:
    line = json.loads(line)

    video_data = {
        "source": line['source'],
        "type": "conv",
        "id": line['id'],
        "video": line['vid_path'],
        "instruction": None,
        "meta": {
            "bboxes": line['box'],
        },
        "conversations": [
            {
                "from": "human",
                "value": "<video>\n" + line['object_class'],
            },
            {
                "from": "gpt",
                "value": line['object_class'] + ": <bboxes>",
            }
        ],
    }
    new_json.append(video_data)

with open(new_file, 'w') as output_file:
    json.dump(new_json, output_file)
