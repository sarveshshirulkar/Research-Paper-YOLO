import os
import cv2
import numpy as np
from scipy.spatial import distance as dist

# Change to the Traffic Signal directory
os.chdir(r'C:\Users\SARVESH\OneDrive\Desktop\Codezz\Python\Traffic Signal')

# Check current working directory
print("Current working directory:", os.getcwd())

# Load YOLO model
net = cv2.dnn.readNet(
    r'C:\Users\SARVESH\OneDrive\Desktop\Codezz\Python\Traffic Signal\yolov4.weights',
    r'C:\Users\SARVESH\OneDrive\Desktop\Codezz\Python\Traffic Signal\yolov4.cfg'
)

# Load COCO dataset class names
with open(r'C:\Users\SARVESH\OneDrive\Desktop\Codezz\Python\Traffic Signal\coco.names', 'r') as f:
    classes = f.read().strip().split('\n')

# Specify the classes we are interested in (cars, bikes, trucks)
vehicle_classes = ['car', 'motorbike', 'truck']

# Get the indices of car, bike, and truck in the class list
car_idx = classes.index('car')
bike_idx = classes.index('motorbike')
truck_idx = classes.index('truck')

# Set up the video capture from a video file
cap = cv2.VideoCapture('vid2.webm')  # Replace with your video file path

# YOLOv4 parameters
layer_names = net.getLayerNames()
output_layers = net.getUnconnectedOutLayers()

if len(output_layers.shape) == 1:  # 1D array
    output_layers = [layer_names[i - 1] for i in output_layers]
else:  # 2D array
    output_layers = [layer_names[i[0] - 1] for i in output_layers]

print("Unconnected Out Layers:", output_layers)

# Store the centroids of detected vehicles
tracked_vehicles = {}

# Function to count vehicles in each frame
def get_vehicle_count(frame):
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward(output_layers)

    current_centroids = []
    car_count = bike_count = truck_count = 0

    # Process the detections
    for output in detections:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:  # Confidence threshold
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # Calculate the centroid
                centroid = (center_x, center_y)
                current_centroids.append((centroid, class_id))

                # Count vehicles based on their class
                if class_id == car_idx:
                    car_count += 1
                elif class_id == bike_idx:
                    bike_count += 1
                elif class_id == truck_idx:
                    truck_count += 1

                # Draw bounding boxes with class names
                color = (0, 255, 0)
                if class_id == car_idx:
                    color = (0, 255, 255)  # Yellow for cars
                elif class_id == bike_idx:
                    color = (255, 0, 0)  # Blue for bikes
                elif class_id == truck_idx:
                    color = (255, 0, 255)  # Magenta for trucks

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                label = f"{classes[class_id]}: {confidence:.2f}"
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Track vehicles by matching current centroids to tracked ones
    for idx, (centroid, class_id) in enumerate(current_centroids):
        if idx not in tracked_vehicles:
            tracked_vehicles[idx] = centroid
        else:
            # Calculate the distance to existing centroids
            dists = dist.cdist([centroid], list(tracked_vehicles.values()), metric="euclidean")[0]
            min_dist_idx = np.argmin(dists)

            # If the minimum distance is small enough, consider it the same vehicle
            if dists[min_dist_idx] < 50:  # Adjust this threshold as needed
                tracked_vehicles[min_dist_idx] = centroid

    total_count = len(tracked_vehicles)
    return car_count, bike_count, truck_count, total_count

# Read and process video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    car_count, bike_count, truck_count, total_count = get_vehicle_count(frame)

    # Display the counts on the video frame
    cv2.putText(frame, f"Cars: {car_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Bikes: {bike_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Trucks: {truck_count}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Total: {total_count}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Vehicle Count', frame)

    # Exit if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
