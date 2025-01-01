import os
import cv2
import numpy as np

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
cap3 = cv2.VideoCapture('vid5.webm')
cap4 = cv2.VideoCapture('vid4.webm')

# YOLO output layers
layer_names = net.getLayerNames()
output_layers = net.getUnconnectedOutLayers()

if len(output_layers.shape) == 1:
    output_layers = [layer_names[i - 1] for i in output_layers]
else:
    output_layers = [layer_names[i[0] - 1] for i in output_layers]

# Tracking dictionary and vehicle ID counter for each frame
tracked_vehicles = {'side1': {}, 'side2': {}, 'side3': {}, 'side4': {}}
next_vehicle_id = {'side1': 0, 'side2': 0, 'side3': 0, 'side4': 0}
unique_vehicle_count = {'side1': 0, 'side2': 0, 'side3': 0, 'side4': 0}
pcu_weighted_count = {'side1': 0, 'side2': 0, 'side3': 0, 'side4': 0}  # PCU-based counts

# Function to assign a unique ID to a new vehicle for each frame
def assign_vehicle_id(side):
    vehicle_id = next_vehicle_id[side]
    next_vehicle_id[side] += 1
    return vehicle_id

# Define bounding boxes for roads in each frame
road_boxes = {
    'side1': [(110, 80), (430, 80), (750, 330), (5, 330)],
    'side2': [(150, 100), (400, 100), (600, 330), (5, 330)],
    'side3': [(230, 100), (730, 100), (880, 480), (100, 480)],
    'side4': [(270, 300), (530, 300), (650, 500), (180, 500)]
}

# Function to check if the vehicle is within the road bounding box
def is_in_road_box(centroid, road_box):
    points = np.array(road_box, dtype=np.int32)
    return cv2.pointPolygonTest(points, centroid, False) >= 0

# Function to count vehicles and assign vehicle IDs with PCU weights
def get_vehicle_count(frame, road_box, side):
    global tracked_vehicles, pcu_weighted_count
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward(output_layers)

    current_centroids = []

    # Reset PCU-weighted count for the current frame
    pcu_weighted_count[side] = 0

    for output in detections:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.4 and class_id in class_indices.values():
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                centroid = (center_x, center_y)

                if is_in_road_box(centroid, road_box):
                    current_centroids.append((centroid, class_id))

                    # Draw bounding boxes and labels
                    color = (0, 255, 0)  # Default green
                    if class_id == class_indices['car']:
                        color = (0, 255, 255)  # Yellow for cars
                    elif class_id == class_indices['motorbike']:
                        color = (255, 0, 0)  # Blue for bikes
                    elif class_id == class_indices['truck']:
                        color = (255, 0, 255)  # Magenta for trucks

                    # Draw bounding box around detected object
                    cv2.rectangle(frame, (center_x - 15, center_y - 15), (center_x + 15, center_y + 15), color, 2)
                    label = f"{classes[class_id]}: {confidence:.2f}"
                    cv2.putText(frame, label, (center_x - 15, center_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    # Match or assign vehicle ID
                    matched = False
                    for vehicle_id, (last_centroid, last_class_id) in tracked_vehicles[side].items():
                        if np.linalg.norm(np.array(centroid) - np.array(last_centroid)) < 50:
                            tracked_vehicles[side][vehicle_id] = (centroid, class_id)
                            matched = True
                            break

                    if not matched:
                        vehicle_id = assign_vehicle_id(side)
                        tracked_vehicles[side][vehicle_id] = (centroid, class_id)
                        unique_vehicle_count[side] += 1

                    # Add PCU weight for detected vehicle type
                    vehicle_type = list(class_indices.keys())[list(class_indices.values()).index(class_id)]
                    pcu_weighted_count[side] += vehicle_classes[vehicle_type]

    # Draw road bounding box
    points = np.array(road_box, dtype=np.int32)
    cv2.polylines(frame, [points], isClosed=True, color=(0, 0, 255), thickness=2)

    return unique_vehicle_count[side]

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

    # Process every 4th frame
    if frame_count % 10 == 0:
        vehicle_count1 = get_vehicle_count(frame1, road_boxes['side1'], 'side1')
        vehicle_count2 = get_vehicle_count(frame2, road_boxes['side2'], 'side2')
        vehicle_count3 = get_vehicle_count(frame3, road_boxes['side3'], 'side3')
        vehicle_count4 = get_vehicle_count(frame4, road_boxes['side4'], 'side4')

        green_signal_time1 = update_green_signal_time(vehicle_count1)
        green_signal_time2 = update_green_signal_time(vehicle_count2)
        green_signal_time3 = update_green_signal_time(vehicle_count3)
        green_signal_time4 = update_green_signal_time(vehicle_count4)

        # Display vehicle count and green time on the top-left corner of each frame
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame1, f"Vehicles: {vehicle_count1}", (150, 30), font, 0.7, (0, 255, 0), 2)
        cv2.putText(frame1, f"Green Time: {green_signal_time1}", (110, 60), font, 0.7, (0, 255, 0), 2)

        cv2.putText(frame2, f"Vehicles: {vehicle_count2}", (140, 30), font, 0.7, (0, 255, 0), 2)
        cv2.putText(frame2, f"Green Time: {green_signal_time2}", (110, 60), font, 0.7, (0, 255, 0), 2)

        cv2.putText(frame3, f"Vehicles: {vehicle_count3}", (190, 30), font, 1.2, (0, 255, 0), 3)
        cv2.putText(frame3, f"Green Time: {green_signal_time3}", (110, 70), font, 1.2, (0, 255, 0), 3)

        cv2.putText(frame4, f"Vehicles: {vehicle_count4}", (270, 40), font, 1.0, (0, 255, 0), 3)
        cv2.putText(frame4, f"Green Time: {green_signal_time4}", (170, 80), font, 1.1, (0, 255, 0), 3)

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
