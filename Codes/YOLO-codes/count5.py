import os
import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

# Change to the Traffic Signal directory
os.chdir(r'C:\Users\SARVESH\OneDrive\Desktop\Codezz\Python\Traffic Signal')

# Load YOLO model
net = cv2.dnn.readNet(
    r'C:\Users\SARVESH\OneDrive\Desktop\Codezz\Python\Traffic Signal\yolov4.weights',
    r'C:\Users\SARVESH\OneDrive\Desktop\Codezz\Python\Traffic Signal\yolov4.cfg'
)

# Load COCO dataset class names
with open(r'C:\Users\SARVESH\OneDrive\Desktop\Codezz\Python\Traffic Signal\coco.names', 'r') as f:
    classes = f.read().strip().split('\n')

# Specify the classes we are interested in (cars, bikes, trucks) and their PCU weights
vehicle_classes = {'car': 1.0, 'motorbike': 0.75, 'truck': 3.7}
class_indices = {cls_name: classes.index(cls_name) for cls_name in vehicle_classes.keys()}

# Set up the video capture from four video files
cap1 = cv2.VideoCapture('vid7.webm')
cap2 = cv2.VideoCapture('vid2.webm')
cap3 = cv2.VideoCapture('vid3.webm')
cap4 = cv2.VideoCapture('vid4.webm')

# YOLO output layers
layer_names = net.getLayerNames()
output_layers = net.getUnconnectedOutLayers()

if len(output_layers.shape) == 1:
    output_layers = [layer_names[i - 1] for i in output_layers]
else:
    output_layers = [layer_names[i[0] - 1] for i in output_layers]

# Deep SORT tracker initialization for each frame
tracker1 = DeepSort(max_age=5)
tracker2 = DeepSort(max_age=5)
tracker3 = DeepSort(max_age=5)
tracker4 = DeepSort(max_age=5)

# Define bounding boxes for roads in each frame
road_boxes = {
    'side1': [(110, 80), (430, 80), (750, 330), (5, 330)],
    'side2': [(150, 100), (400, 100), (600, 330), (5, 330)],
    'side3': [(180, 150), (480, 150), (580, 330), (140, 330)],
    'side4': [(270, 300), (530, 300), (650, 500), (180, 500)]
}

# Function to check if the vehicle is within the road bounding box
def is_in_road_box(centroid, road_box):
    points = np.array(road_box, dtype=np.int32)
    return cv2.pointPolygonTest(points, centroid, False) >= 0

# Function to process each frame with Deep SORT and PCU weighting
def process_frame_with_tracking(frame, road_box, side, tracker):
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward(output_layers)

    detections_for_tracker = []  # List for Deep SORT input
    pcu_weighted_count = 0  # PCU-based vehicle count

    for output in detections:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.4 and class_id in class_indices.values():
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                centroid = (center_x, center_y)

                # Check if the vehicle is within the defined road box
                if is_in_road_box(centroid, road_box):
                    bbox = [center_x - w // 2, center_y - h // 2, w, h]
                    detections_for_tracker.append((bbox, confidence, class_id))

                    # Increment PCU count based on vehicle class
                    vehicle_type = list(class_indices.keys())[list(class_indices.values()).index(class_id)]
                    pcu_weighted_count += vehicle_classes[vehicle_type]

    # Update Deep SORT tracker
    tracks = tracker.update_tracks(detections_for_tracker, frame=frame)

    # Draw bounding boxes, labels, and IDs for tracked vehicles
    for track in tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue
        track_id = track.track_id
        bbox = track.to_ltrb()
        x1, y1, x2, y2 = map(int, bbox)
        class_id = track.get_class()

        # Determine color by class
        color = (0, 255, 255) if class_id == class_indices['car'] else (255, 0, 0) if class_id == class_indices['motorbike'] else (255, 0, 255)
        label = f"{classes[class_id]} ID: {track_id}"

        # Draw bounding box and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Draw road bounding box
    points = np.array(road_box, dtype=np.int32)
    cv2.polylines(frame, [points], isClosed=True, color=(0, 0, 255), thickness=2)

    return pcu_weighted_count

# Function to update green signal time based on PCU-weighted count
def update_green_signal_time(pcu_count, base_time=10, extra_time_per_unit=2, max_time=90, min_time=10):
    green_signal_time = base_time + (pcu_count * extra_time_per_unit)
    return max(min_time, min(int(green_signal_time), max_time))

# Function to resize and combine frames into a 2x2 matrix
def resize_and_concatenate_frames(frames, target_height=480, target_width=640, scale=0.7):
    resized_frames = [cv2.resize(frame, (target_width, target_height)) for frame in frames]
    top_row = np.hstack((resized_frames[0], resized_frames[1]))
    bottom_row = np.hstack((resized_frames[2], resized_frames[3]))
    matrix_frame = np.vstack((top_row, bottom_row))

    # Reduce the final output by 30%
    new_size = (int(matrix_frame.shape[1] * scale), int(matrix_frame.shape[0] * scale))
    matrix_frame = cv2.resize(matrix_frame, new_size)
    
    return matrix_frame

# Process the video streams and display frames
frame_count = 0
while cap1.isOpened() and cap2.isOpened() and cap3.isOpened() and cap4.isOpened():
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    ret3, frame3 = cap3.read()
    ret4, frame4 = cap4.read()
    

    if not (ret1 and ret2 and ret3 and ret4):
        break

    # Process every 10th frame
    if frame_count % 10 == 0:
        pcu_count1 = process_frame_with_tracking(frame1, road_boxes['side1'], 'side1', tracker1)
        pcu_count2 = process_frame_with_tracking(frame2, road_boxes['side2'], 'side2', tracker2)
        pcu_count3 = process_frame_with_tracking(frame3, road_boxes['side3'], 'side3', tracker3)
        pcu_count4 = process_frame_with_tracking(frame4, road_boxes['side4'], 'side4', tracker4)

        green_signal_time1 = update_green_signal_time(pcu_count1)
        green_signal_time2 = update_green_signal_time(pcu_count2)
        green_signal_time3 = update_green_signal_time(pcu_count3)
        green_signal_time4 = update_green_signal_time(pcu_count4)

        font = cv2.FONT_HERSHEY_SIMPLEX
        #cv2.putText(frame1, f"PCU Count: {pcu_count1:.2f}", (150, 30), font, 0.7, (0, 0, 0), 2)
        #cv2.putText(frame1, f"Green Time: {green_signal_time1}", (110, 60), font, 0.7, (0, 0, 0), 2)

        cv2.putText(frame1, f"Vehicles: {pcu_count1:.2f}", (150, 30), font, 0.7, (0, 0, 0), 2)
        cv2.putText(frame1, f"Green Time: {green_signal_time1}", (110, 60), font, 0.7, (0, 0, 0), 2)

        cv2.putText(frame2, f"Vehicles: {pcu_count2:.2f}", (140, 30), font, 0.7, (0, 0, 0), 2)
        cv2.putText(frame2, f"Green Time: {green_signal_time2}", (110, 60), font, 0.7, (0, 0, 0), 2)

        cv2.putText(frame3, f"Vehicles: {pcu_count3:.2f}", (150, 30), font, 0.7, (0, 0, 0), 2)
        cv2.putText(frame3, f"Green Time: {green_signal_time3}", (110, 60), font, 0.7, (0, 0, 0), 2)

        cv2.putText(frame4, f"Vehicles: {pcu_count4:.2f}", (470, 40), font, 1.0, (0, 0, 0), 3)
        cv2.putText(frame4, f"Green Time: {green_signal_time4}", (380, 80), font, 1.1, (0, 0, 0), 3)

        # Combine and display the frames in a 2x2 matrix layout
        combined_frame = resize_and_concatenate_frames([frame1, frame2, frame3, frame4])
        cv2.imshow('Traffic Signal Monitoring', combined_frame)

        # Break on pressing the 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    frame_count += 1

# Release video captures and close windows
cap1.release()
cap2.release()
cap3.release()
cap4.release()
cv2.destroyAllWindows()