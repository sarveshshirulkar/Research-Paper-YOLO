import torch
import cv2
import numpy as np
import os

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # You can choose 'yolov5s', 'yolov5m', 'yolov5l', or 'yolov5x'

# Specify the classes we are interested in (cars, bikes, trucks)
vehicle_classes = ['car', 'motorbike', 'truck']

# Set up the video capture from a video file
cap = cv2.VideoCapture('vid2.webm')  # Replace with your video file path

# Tracking dictionary and vehicle ID counter
tracked_vehicles = {}
next_vehicle_id = 0

# Function to assign a unique ID to a new vehicle
def assign_vehicle_id():
    global next_vehicle_id
    vehicle_id = next_vehicle_id
    next_vehicle_id += 1
    return vehicle_id

# Function to count vehicles in each frame
def get_vehicle_count(frame):
    global tracked_vehicles
    
    # Run YOLOv5 inference
    results = model(frame)

    # Process detections
    detections = results.xyxy[0].numpy()  # Get results as numpy array

    current_centroids = []
    class_counts = {'car': 0, 'motorbike': 0, 'truck': 0}

    for detection in detections:
        x1, y1, x2, y2, conf, class_id = detection
        class_name = results.names[int(class_id)]
        
        # We're only interested in cars, bikes, and trucks
        if class_name in vehicle_classes and conf > 0.5:
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            centroid = (center_x, center_y)
            current_centroids.append((centroid, class_name))
            
            # Draw bounding boxes with class names
            color = (0, 255, 0)
            if class_name == 'car':
                color = (0, 255, 255)  # Yellow for cars
            elif class_name == 'motorbike':
                color = (255, 0, 0)  # Blue for bikes
            elif class_name == 'truck':
                color = (255, 0, 255)  # Magenta for trucks

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            label = f"{class_name}: {conf:.2f}"
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Increment class count
            class_counts[class_name] += 1

    # Tracking vehicles
    for idx, (centroid, class_name) in enumerate(current_centroids):
        matched = False
        for vehicle_id, (last_centroid, last_class_name) in tracked_vehicles.items():
            if np.linalg.norm(np.array(centroid) - np.array(last_centroid)) < 50:  # Threshold for matching
                tracked_vehicles[vehicle_id] = (centroid, class_name)  # Update position
                matched = True
                break

        if not matched:  # New vehicle detected
            vehicle_id = assign_vehicle_id()
            tracked_vehicles[vehicle_id] = (centroid, class_name)

    # Update vehicle count
    vehicle_count = len(tracked_vehicles)

    # Draw vehicle IDs
    for vehicle_id, (centroid, class_name) in tracked_vehicles.items():
        cv2.putText(frame, f"ID: {vehicle_id}", (centroid[0] + 10, centroid[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Return total vehicle count and individual class counts
    return vehicle_count, class_counts['car'], class_counts['motorbike'], class_counts['truck']


# Function to update green signal time based on vehicle count
def update_green_signal_time(car_count, bike_count, truck_count, base_time=10, max_time=90, min_time=10):
    """
    Calculate the green signal time based on the number of vehicles.
    
    Parameters:
    - car_count, bike_count, truck_count: Count of each vehicle type.
    - base_time: Base green signal time in seconds.
    
    Returns:
    - green_signal_time (int): Updated green signal time in seconds.
    """
    additional_time = (bike_count * 0.25) + (car_count * 0.5) + (truck_count * 0.75)
    green_signal_time = base_time + additional_time
    return max(min_time, min(green_signal_time, max_time))


# Process video frames and update the green signal time
frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame from video.")

    # Process every 4th frame
    if frame_count % 4 == 0:
        vehicle_count, car_count, bike_count, truck_count = get_vehicle_count(frame)
        
        # Display vehicle counts on the frame
        cv2.putText(frame, f"Total Vehicles: {vehicle_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # Update and display green signal time
        green_signal_time = update_green_signal_time(car_count, bike_count, truck_count)
        cv2.putText(frame, f"Green Signal Time: {green_signal_time}s", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show the frame with vehicle counts and signal time
    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.imshow('Test Window', blank_frame)
    cv2.waitKey(0)  # Wait indefinitely until a key is pressed

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1

# Release resources
cap.release()
cv2.destroyAllWindows()
