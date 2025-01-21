import json
import re


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


# Used to convert frame ids to normalized counts.
def convert(duration, x, num_frames=100):
    x = x / duration * num_frames
    x = str(min(round(x), num_frames - 1))
    # TODO: need to adjust if we ever go above 100 frames.
    if len(x) == 1:
        x = "0" + x
    return x


def main():
    dataset = json.load(open('conv_vidstg-it_train.json', 'r'))
    print(len(dataset))
    num_frames = 100

    row = dataset[2]
    print(row)

    # Print the raw conversation.
    for step in row['conversations']:
        print(step)

    if 'meta' in row:
        # Replace the tokens with normalized frame ids.
        replace_set = []
        for k, v in row['meta']['token'].items():
            replace_set.append((k, convert(row['meta']['duration'], v, num_frames)))
        for l in range(len(row['conversations'])):
            for x1, x2 in replace_set:
                row['conversations'][l]['value'] = row['conversations'][l]['value'].replace(x1, x2)
        # Replace bounding boxes.
        all_bbox_fid = list(row['meta']['bboxes'].keys())
        all_norm_fid = [x[1] for x in replace_set]
        all_norm_fid = [str(x) for x in range(int(all_norm_fid[0]), int(all_norm_fid[1]) + 1)]
        num_bbox = len(all_bbox_fid)
        num_norm = len(all_norm_fid)
        selected_bbox_fid = [x for x in range(int(all_bbox_fid[0]), int(all_bbox_fid[-1]) + 1, num_bbox // (num_norm - 1))]
        normalize_frame_to_bbox = {}
        for fid, norm_fid in zip(selected_bbox_fid, all_norm_fid):
            normalize_frame_to_bbox[int(norm_fid)] = row['meta']['bboxes'][str(fid)]
        # bbox_string = str(normalize_frame_to_bbox)
        bbox_string = re.sub(r'\s+', '', str(normalize_frame_to_bbox))
        row['conversations'][l]['value'] = row['conversations'][l]['value'].replace('<bboxes>', bbox_string)

    # Print the conversation after processing.
    for step in row['conversations']:
        print(step)

    # Visualize the video.
    import cv2

    video_path = row['video']
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file {video_path}")
        exit()

    all_bbox_fids = [int(x) for x in list(row['meta']['bboxes'].keys())]
    # This selects based on frame_ids in the prompt. There is probably a better way to do this.
    fid_dict = {k: v for k, v in zip(selected_bbox_fid, all_norm_fid)}
    all_bbox_fids = [x for x in all_bbox_fids if x in fid_dict]

    frame_count = 0
    while True:
        frame_count += 1
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count in all_bbox_fids:
            bbox = row['meta']['bboxes'][str(frame_count)]
            bbox = unnormalize_bbox(bbox, frame.shape[1], frame.shape[0])
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 1) 
        cv2.imshow('Frame', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
