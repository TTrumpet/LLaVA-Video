import json
import random

file_name = "/lustre/fs1/home/ti727611/CAP/LLaVA-Video/stvg.json"
new_file = "conv_stvg2.json"
new_json = []

data_path = "/datasets/vidor/video/"

with open(file_name, 'r') as f:
    data = json.load(f)

for line in data:

    new_conversations = []
    new_input_value = line["conversations"][0]["value"]
    new_output_value = line["conversations"][1]["value"]

    temp_inputs = line["variables"]["temporal_input_locations"]
    width_inputs = line["variables"]["spatial_width_input_locations"]
    height_inputs = line["variables"]["spatial_height_input_locations"]

    temp_outputs = line["variables"]["temporal_output_locations"]
    width_outputs = line["variables"]["spatial_width_output_locations"]
    height_outputs = line["variables"]["spatial_height_output_locations"]

    counter = 0
    while "<TEMP-OUTPUT>" in new_output_value:
        if counter == 0 or counter == len(temp_outputs) - 2:
            new_output_value = new_output_value.replace("<TEMP-OUTPUT>", str(temp_outputs[counter]) + ", ", 1)
        else:
            new_output_value = new_output_value.replace("<TEMP-OUTPUT>", str(temp_outputs[counter]), 1)
        counter += 1

    counter = 0
    while "<WIDTH-OUTPUT>" in new_output_value:
        new_output_value = new_output_value.replace("<WIDTH-OUTPUT>", str(width_outputs[counter]) + ", ", 1)
        counter += 1


    counter = 0
    while "<HEIGHT-OUTPUT>" in new_output_value:
        if counter % 2 == 0:
            new_output_value = new_output_value.replace("<HEIGHT-OUTPUT>", str(height_outputs[counter]) + ", ", 1)

        else:
            new_output_value = new_output_value.replace("<HEIGHT-OUTPUT>", str(height_outputs[counter]), 1)
        counter += 1

    counter = 0
    while "<TEMP-INPUT>" in new_input_value:
        if counter == 0 or counter == len(temp_outputs) - 2:
            new_input_value = new_input_value.replace("<TEMP-INPUT>", str(temp_inputs[counter]) + ", ", 1)
        else:
            new_input_value = new_input_value.replace("<TEMP-INPUT>", str(temp_inputs[counter]), 1)
        counter += 1

    counter = 0
    while "<WIDTH-INPUT>" in new_input_value:
        new_input_value = new_input_value.replace("<WIDTH-INPUT>", str(width_inputs[counter]) + ", ", 1)
        counter += 1

    counter = 0
    while "<HEIGHT-INPUT>" in new_input_value:
        if counter % 2 == 0:
            new_input_value = new_input_value.replace("<HEIGHT-INPUT>", str(height_inputs[counter]) + ", ", 1)

        else:
            new_input_value = new_input_value.replace("<HEIGHT-INPUT>", str(height_inputs[counter]), 1)
        counter += 1


    new_conversations.append(
            {
                "role": "human",
                "content": new_input_value
            }
    )

    new_conversations.append(
            {
                "role": "gpt",
                "content": new_output_value
            }
    )


    video_data = {
            "source": None,
            "type": line["type"],
            "id": line["id"],
            "video": data_path + line["video"], # path
            "instruction": None,
            "meta": line["meta"],
            "messages": new_conversations
    } 

    new_json.append(video_data)

with open(new_file, 'w') as output_file:
    json.dump(new_json, output_file)
