import json

def compute_iou(pred_start, pred_end, gt_start, gt_end):
    """
    Computes the temporal Intersection over Union (IoU)
    between a predicted segment and a ground truth segment.
    """
    # Calculate intersection length
    intersection = max(0, min(pred_end, gt_end) - max(pred_start, gt_start))
    # Calculate union length
    union = max(pred_end, gt_end) - min(pred_start, gt_start)
    return intersection / union if union > 0 else 0

def compute_iop(pred_start, pred_end, gt_start, gt_end):
    """
    Computes the temporal Intersection over Prediction (IoP)
    between a predicted segment and a ground truth segment.
    """
    intersection = max(0, min(pred_end, gt_end) - max(pred_start, gt_start))
    pred_duration = pred_end - pred_start
    return intersection / pred_duration if pred_duration > 0 else 0

# Compute IoP @ 0.5
def compute_iop_05(pred_start, pred_end, gt_start, gt_end):
    """
    Computes the temporal Intersection over Prediction (IoP)
    between a predicted segment and a ground truth segment.
    """
    intersection = max(0, min(pred_end, gt_end) - max(pred_start, gt_start))
    pred_duration = pred_end - pred_start
    return intersection / pred_duration if pred_duration > 0.5 else 0

def main():
    json_file = 'evaluation/vtimellm_results.json'

    # Load the JSON data from file
    with open(json_file, "r") as f:
        results = json.load(f)

    all_iou = []
    all_iop = []
    all_iop_05 = []

    # Process each video and its corresponding segments
    for video, segments in results.items():
        print(f"Video: {video}")
        for idx, entry in enumerate(segments):
            try:
                pred_start = int(entry["pred_start"])
                pred_end = int(entry["pred_end"])
                gt_start = int(entry["gt_start"])
                gt_end = int(entry["gt_end"])

                iou = compute_iou(pred_start, pred_end, gt_start, gt_end)
                iop = compute_iop(pred_start, pred_end, gt_start, gt_end)
                iop_05 = compute_iop_05(pred_start, pred_end, gt_start, gt_end)

                all_iou.append(iou)
                all_iop.append(iop)
                all_iop_05.append(iop_05)

                print(f"  Entry {idx+1}: IoU = {iou:.4f}, IoP = {iop:.4f}")
            except:
                pass

    # Compute and display average metrics
    avg_iou = sum(all_iou) / len(all_iou) if all_iou else 0
    avg_iop = sum(all_iop) / len(all_iop) if all_iop else 0
    avg_iop_05 = sum(all_iop_05) / len(all_iop_05) if all_iop_05 else 0

    print("\nOverall Metrics:")
    print(f"  Average IoU: {avg_iou:.4f}")
    print(f"  Average IoP: {avg_iop:.4f}")
    print(f"  Average IoP @ 0.5: {avg_iop_05:.4f}")

if __name__ == "__main__":
    main()
