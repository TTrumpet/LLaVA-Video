import cv2
import json
new_file = "conv_ElysiumTrack-100K-Newton.json"
new_json = []

f = open("/lustre/fs1/home/jfioresi/datasets/elysium_track/ElysiumTrack-1M-Newton.json", 'r')
output_file = open(new_file, 'w')

#data = json.load(f)
counter = 0
for line in f:
    line = json.loads(line)
    print(line)

    # convert bounding boxes to same format as vidshikra:
    # {frame_number: [bbbox], ...}
    # not uniform sampling -> use given frames .jpg 

    # read video to find out fps of frames

    bboxes = line['box']
    video_path = line['vid_path']

    cap = cv2.VideoCapture(video_path)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(counter)
    
    print(num_frames)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print(fps)
    print(len(line['frames']))
    print(len(bboxes))

    # get frame ids
    index = 0
    inc = num_frames / fps
    new_bbox = {}
    for bbox in bboxes:
        new_bbox[int(index)] = bbox
        index += inc

    print(new_bbox)
    print(len(new_bbox))


    video_data = {
            "source": line['source'],
            "type": "conv",
            "id": line['id'],
            "video": video_path,
            "instruction": None,
            "conversations": [
                {
                    "from": "human",
                    "value": "<video>\n" + line['object_class'],
                },
                {
                    "from": "gpt",
                    "value": new_bbox,
                }
            ],
        }
    new_json.append(video_data)
    counter += 1
    if counter == 2:
        exit()
    if counter == 100000:
        break
json.dump(new_json, output_file)
