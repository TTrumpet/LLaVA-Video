import json
new_file = "conv_100_vidstg-it_train.json"
new_json = []

f = open("/lustre/fs1/home/ttran/CAP/LLaVA-Video/vidstg_train.json", 'r')
output_file = open(new_file, 'w')
data = json.load(f)

for line in data:
    '''
    tokens = line['meta']['token']
    bboxes = str(line['meta']['bboxes'])
    new_conv = []
    for item in line['conversations']:
        new_item = item
        for f, v in item.items():
            for token, val in tokens.items():
                new_item[f] = new_item[f].replace(token, str(val))
            new_item[f] = new_item[f].replace('<bboxes>', str(bboxes))
        new_conv.append(new_item)
    '''
    dirs = line['meta']['vid_path'].split("/")
    new_vid_path = '/datasets/vidor/video/' + dirs[len(dirs) - 2] + "/"  + dirs [len(dirs) - 1]

    video_data = {
            "source": "vidstg",
            "type": "conv",
            "id": line['id'],
            #"video": '/datasets/' + line['video'],
            "video": new_vid_path,
            "instruction": None,
            #"conversations": new_conv,
            "conversations": line['conversations'],
        }
    new_json.append(video_data)
json.dump(new_json, output_file)
