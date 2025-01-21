import ast
import cv2
import imageio
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import os
import json

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

#TODO: display using matplotlib and make saving optional
# visualize with actual bboxes
def visualize_output(video_path, text_output, model_name='vidshikra', split='train', save=True, show_actual=False):

    video_name = os.path.basename(os.path.normpath(video_path)).strip(".mp4")
    output = text_output[text_output.index("{"):text_output.index("}") + 1]
    frames = ast.literal_eval(output)

    data = None
    actual_bbox = None
    if show_actual:
        with open(f'/lustre/fs1/home/jfioresi/datasets/shikra_v/annotations/vidstg-it_{split}.json') as json_data:
            data = json.load(json_data)
        for video in data:
            if os.path.basename(os.path.normpath(video['video'])).strip(".mp4") == video_name:
                actual_bbox = video['meta']['bboxes']
        #print(actual_bbox)

    video = []
    print(video_path)
    cap = cv2.VideoCapture(video_path)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(num_frames)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(height)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print(width)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print(fps)
    frame_index = 1

    print('reading ' + video_name + '.mp4 ...')
    while(frame_index <= num_frames):
        ret, frame = cap.read()
        #while not ret:
        #    ret, frame = cap.read()
        try:
            bbox = frames[int(frame_index / 64 * 100)]
        except:
            bbox = frames[len(frames) - 1]
        coords = unnormalize_bbox(bbox, width, height)
        x1 = coords[0]
        y1 = coords[1]
        x2 = coords[2]
        y2 = coords[3]
        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2) # blue

        if show_actual:
            if str(frame_index) in actual_bbox.keys():
                actual = actual_bbox[str(frame_index)]
                actual_coords = unnormalize_bbox(actual, width, height)
                x1 = actual_coords[0]
                y1 = actual_coords[1]
                x2 = actual_coords[2]
                y2 = actual_coords[3]
                frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) # green
            # else not in frame
            else:
                video.append(frame)
                frame_index += 1
                continue

        #plt.imshow(frame)
        #plt.axis('off')
        #plt.draw()
        #plt.show()
        video.append(frame)
        frame_index += 1
    cap.release()

    vid_frames = np.array(video)
    kwargs = {'macro_block_size': None}
    if show_actual:
        model_name += '_two'
    try:
        imageio.mimwrite(video_name + '_' + model_name + '.mp4', vid_frames[:, :, :, ::-1], fps=fps, **kwargs)
    except:
        print('failed')
        return 'Failed'
    return vid_frames

if __name__ == '__main__':
    #video_path = "/lustre/fs1/home/ttran/CAP/LLaVA-Video/1014386846.mp4"
    #text_outputs = "{0:[3,12,45,98],1:[3,12,46,97],2:[3,12,46,97],3:[3,12,46,97],4:[3,12,46,97],5:[3,12,46,97],6:[3,12,46,97],7:[3,12,46,97],8:[3,12,46,97],9:[3,12,46,97],10:[3,12,46,97],11:[3,12,46,97],12:[3,12,46,97],13:[3,12,46,97],14:[3,12,46,97],15:[3,12,46,97],16:[3,12,46,97],17:[3,12,46,97],18:[3,12,46,97],19:[3,12,46,97],20:[3,12,46,97],21:[3,12,46,97],22:[3,12,46,97],23:[3,12,46,97],24:[3,12,46,97],25:[3,12,46,97],26:[3,12,46,97],27:[3,12,46,97],28:[3,12,46,97],29:[3,12,46,97],30:[3,12,46,97],31:[3,12,46,97],32:[3,12,46,97],33:[3,12,46,97],34:[3,12,46,97],35:[3,12,46,97],36:[3,12,46,97],37:[3,12,46,97],38:[3,12,46,97],39:[3,12,46,97],40:[3,12,46,97],41:[3,12,46,97],42:[3,12,46,97],43:[3,12,46,97],44:[3,12,46,97],45:[3,12,46,97],46:[3,12,46,97],47:[3,12,46,97],48:[3,12,46,97],49:[3,12,46,97],50:[3,12,46,97],51:[3,12,46,97],52:[3,12,46,97],53:[3,12,46,97],54:[3,12,46,97],55:[3,12,46,97],56:[3,12,46,97],57:[3,12,46,97],58:[3,12,46,97],59:[3,12,46,97],60:[3,12,46,97],61:[3,12,46,97],62:[3,12,46,97],63:[3,12,46,97],64:[3,12,46,97],65:[3,12,46,97],66:[3,12,46,97],67:[3,12,46,97],68:[3,12,46,97],69:[3,12,46,97],70:[3,12,46,97],71:[3,12,46,97],72:[3,12,46,97],73:[3,12,46,97],74:[3,12,46,97],75:[3,12,46,97],76:[3,12,46,97],77:[3,12,46,97],78:[3,12,46,97],79:[3,12,46,97],80:[3,12,46,97],81:[3,12,46,97],82:[3,12,46,97],83:[3,12,46,97],84:[3,12,46,97],85:[3,12,46,97],86:[3,12,46,97],87:[3,12,46,97],88:[3,12,46,97],89:[3,12,46,97],90:[3,12,46,97],91:[3,12,46,97],92:[3,12,46,97],93:[3,12,46,97],94:[3,12,46,97],95:[3,12,46,97],96:[3,12,46,97],97:[3,12,46,97],98:[3,12,46,97],99:[3,12,46,97]}"

    num_videos = 5
    model_name_short = 'vidshikra_100'
    split='train'
    with open(f'{num_videos}_{model_name_short}_{split}_outputs.json') as json_data:
        data = json.load(json_data)
    for i in data:
        print(f"video #{i}:")
        video_path = os.path.join('/datasets', data[i]['video_path'])
        visualize_output(video_path, data[i]['text_outputs'], model_name=model_name_short, split=split, show_actual=True)
