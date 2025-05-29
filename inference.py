import cv2
import torch
import numpy as np
from collections import deque
from torchvision.ops import nms
from torchvision.transforms import functional as F
from torchvision.models.detection import ssd300_vgg16

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using Device: {device}")

checkpoint = torch.load('ssd_model_one_class.pth', map_location=device)
class_names = checkpoint.get('class_names', ['background', 'non-defect', 'defect'])
num_classes = len(class_names)
model = ssd300_vgg16(weights=None)
model.head.classification_head.num_classes = num_classes

model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

CONFIDENCE_THRESHOLD = 0.7
IOU_THRESHOLD = 0.3
DETECTION_PERSISTENCE = 2 
MAX_HISTORY = 5  
DEBUG_MODE = True 

def preprocess(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (300, 300))
    img = F.to_tensor(img).unsqueeze(0).to(device)
    return img

def apply_nms(boxes, scores, iou_threshold=IOU_THRESHOLD):
    if len(boxes) == 0:
        return torch.tensor([], device=device, dtype=torch.int64)
    keep = nms(boxes, scores, iou_threshold)
    return keep

def draw_boxes(frame, boxes, labels, scores, threshold=CONFIDENCE_THRESHOLD):
    drawn = False
    for box, label, score in zip(boxes, labels, scores):
        if score < threshold:
            continue

        label = label.item()
        if label <= 0 or label >= len(class_names):
            continue

        class_text = class_names[label]

        box = box.int().cpu().numpy()
        x1, y1, x2, y2 = box
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if (x2 - x1) < 20 or (y2 - y1) < 20:
            continue

        color = (0, 255, 0) if label == 1 else (0, 0, 255)
        text = f"{class_text}: {score:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        drawn = True

    if not drawn:
        cv2.putText(frame, "No Objects Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return frame

def box_iou(box1, box2):
    if isinstance(box1, torch.Tensor):
        box1 = box1.cpu().numpy()
    if isinstance(box2, torch.Tensor):
        box2 = box2.cpu().numpy()
        
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    if union <= 0:
        return 0
    
    return intersection / union

def is_consistent_detection(detection_history, current_boxes, current_scores, min_persistence=DETECTION_PERSISTENCE):
    if len(detection_history) < min_persistence - 1:
        return current_boxes, current_scores
    
    consistent_indices = []
    for i, (box, score) in enumerate(zip(current_boxes, current_scores)):
        if score < CONFIDENCE_THRESHOLD:
            continue
            
        consistent_count = 1  
        for past_boxes, _ in detection_history:
            matched = False
            for past_box in past_boxes:
                if box_iou(box, past_box) > 0.5:  
                    matched = True
                    break
            
            if matched:
                consistent_count += 1
        
        if consistent_count >= min_persistence:
            consistent_indices.append(i)

    if not consistent_indices:
        return torch.tensor([], device=device), torch.tensor([], device=device)
    
    consistent_indices = torch.tensor(consistent_indices, device=device)
    return current_boxes[consistent_indices], current_scores[consistent_indices]

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    frame_count = 0
    detection_history = deque(maxlen=MAX_HISTORY)
    
    prev_time = cv2.getTickCount()
    
    print(f"Class Names in Model: {class_names}")
    print(f"Using Confidence Threshold: {CONFIDENCE_THRESHOLD}")
    print(f"Using Persistence Threshold: {DETECTION_PERSISTENCE} frames")
    print(f"Label Mapping: 1 → {class_names[1]} (non-defect), 2 → {class_names[2]} (defect)")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to Read Frame")
                break

            frame_count += 1
            
            current_time = cv2.getTickCount()
            fps = cv2.getTickFrequency() / (current_time - prev_time)
            prev_time = current_time

            orig_h, orig_w = frame.shape[:2]
            input_tensor = preprocess(frame)

            with torch.no_grad():
                outputs = model(input_tensor)[0]

            boxes = outputs['boxes']
            labels = outputs['labels']
            scores = outputs['scores']

            if len(scores) > 0 and frame_count % 30 == 0:  
                top_indices = torch.where(scores > 0.3)[0]
                if len(top_indices) > 0:
                    top_labels_raw = labels[top_indices][:5].tolist()
                    top_labels_mapped = []
                    for l in top_labels_raw:
                        if l == 2:
                            top_labels_mapped.append(f"{l} → {class_names[2]} (defect)")
                        elif l == 1:
                            top_labels_mapped.append(f"{l} → {class_names[1]} (non-defect)")
                        else:
                            top_labels_mapped.append(f"{l} → {class_names[0]} (background)")

            high_conf_mask = scores > (CONFIDENCE_THRESHOLD * 0.8)  
            boxes = boxes[high_conf_mask]
            labels = labels[high_conf_mask]
            scores = scores[high_conf_mask]

            keep = apply_nms(boxes, scores)
            boxes = boxes[keep]
            labels = labels[keep]
            scores = scores[keep]

            if len(boxes) > 0:
                scale_x = orig_w / 300
                scale_y = orig_h / 300
                boxes = boxes * torch.tensor([scale_x, scale_y, scale_x, scale_y], device=device)

            detection_history.append((boxes.clone(), scores.clone()))
            consistent_boxes, consistent_scores = is_consistent_detection(
                detection_history, boxes, scores
            )
            
            if len(consistent_boxes) > 0:
                consistent_indices = [i for i, box in enumerate(boxes) if any(torch.equal(box, cbox) for cbox in consistent_boxes)]
                consistent_labels = labels[torch.tensor(consistent_indices, device=device)]
                frame = draw_boxes(frame, consistent_boxes, consistent_labels, consistent_scores)
            else:
                cv2.putText(frame, "No Consistent Objects Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, orig_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow('SSD Object Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Application Closed")

if __name__ == "__main__":
    main()