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

# Specify the classes we are interested in (cars, bikes, trucks)
vehicle_classes = ['car', 'motorbike', 'truck']
car_idx = classes.index('car')
bike_idx = classes.index('motorbike')
truck_idx = classes.index('truck')

# Set up the video capture from a video file
cap = cv2.VideoCapture('vid3.webm')  # Replace with your video file path

layer_names = net.getLayerNames()
output_layers = net.getUnconnectedOutLayers()

if len(output_layers.shape) == 1:
    output_layers = [layer_names[i - 1] for i in output_layers]
else:
    output_layers = [layer_names[i[0] - 1] for i in output_layers]

# Tracking dictionary and vehicle ID counter
tracked_vehicles = {}
vehicle_count = 0
next_vehicle_id = 0

# Function to assign a unique ID to a new vehicle
def assign_vehicle_id():
    global next_vehicle_id
    vehicle_id = next_vehicle_id
    next_vehicle_id += 1
    return vehicle_id

# Function to count vehicles in each frame
# def get_vehicle_count(frame):
#     global vehicle_count  # Declare vehicle_count as global
#     height, width = frame.shape[:2]
#     blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
#     net.setInput(blob)
#     detections = net.forward(output_layers)

#     current_centroids = []
#     vehicle_ids = []
    
#     # Initialize class counts
#     class_counts = {car_idx: 0, bike_idx: 0, truck_idx: 0}

#     # Process the detections
#     for output in detections:
#         for detection in output:
#             scores = detection[5:]
#             class_id = np.argmax(scores)
#             confidence = scores[class_id]

#             if confidence > 0.5:
#                 center_x = int(detection[0] * width)
#                 center_y = int(detection[1] * height)

#                 centroid = (center_x, center_y)
#                 current_centroids.append((centroid, class_id))

#                 # Draw bounding boxes with class names
#                 color = (0, 255, 0)
#                 if class_id == car_idx:
#                     color = (0, 255, 255)  # Yellow for cars
#                 elif class_id == bike_idx:
#                     color = (255, 0, 0)  # Blue for bikes
#                 elif class_id == truck_idx:
#                     color = (255, 0, 255)  # Magenta for trucks

#                 cv2.rectangle(frame, (center_x - 15, center_y - 15), (center_x + 15, center_y + 15), color, 2)
#                 label = f"{classes[class_id]}: {confidence:.2f}"
#                 cv2.putText(frame, label, (center_x - 15, center_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

#                 # Increment class count
#                 if class_id in class_counts:
#                     class_counts[class_id] += 1

#     # Tracking vehicles
#     for idx, (centroid, class_id) in enumerate(current_centroids):
#         matched = False
#         for vehicle_id, (last_centroid, last_class_id) in tracked_vehicles.items():
#             if np.linalg.norm(np.array(centroid) - np.array(last_centroid)) < 50:  # Threshold for matching
#                 tracked_vehicles[vehicle_id] = (centroid, class_id)  # Update position
#                 vehicle_ids.append(vehicle_id)
#                 matched = True
#                 break

#         if not matched:  # New vehicle detected
#             vehicle_id = assign_vehicle_id()
#             tracked_vehicles[vehicle_id] = (centroid, class_id)
#             vehicle_ids.append(vehicle_id)

#     # Update vehicle count
#     vehicle_count = len(tracked_vehicles)

#     # Count distinct IDs for cars
#     car_count = sum(1 for v_id, (centroid, class_id) in tracked_vehicles.items() if class_id == car_idx)

#     # Draw vehicle IDs
#     for vehicle_id, (centroid, class_id) in tracked_vehicles.items():
#         cv2.putText(frame, f"ID: {vehicle_id}", (centroid[0] + 10, centroid[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

#     return vehicle_count, class_counts, car_count

def get_vehicle_count(frame):
    global vehicle_count  # Declare vehicle_count as global
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward(output_layers)

    current_centroids = []
    vehicle_ids = []
    
    # Initialize class counts
    class_counts = {car_idx: 0, bike_idx: 0, truck_idx: 0}

    # Process the detections
    for output in detections:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)

                centroid = (center_x, center_y)
                current_centroids.append((centroid, class_id))

                # Draw bounding boxes with class names
                color = (0, 255, 0)
                if class_id == car_idx:
                    color = (0, 255, 255)  # Yellow for cars
                elif class_id == bike_idx:
                    color = (255, 0, 0)  # Blue for bikes
                elif class_id == truck_idx:
                    color = (255, 0, 255)  # Magenta for trucks

                cv2.rectangle(frame, (center_x - 15, center_y - 15), (center_x + 15, center_y + 15), color, 2)
                label = f"{classes[class_id]}: {confidence:.2f}"
                cv2.putText(frame, label, (center_x - 15, center_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Increment class count
                if class_id in class_counts:
                    class_counts[class_id] += 1

    # Tracking vehicles
    for idx, (centroid, class_id) in enumerate(current_centroids):
        matched = False
        for vehicle_id, (last_centroid, last_class_id) in tracked_vehicles.items():
            if np.linalg.norm(np.array(centroid) - np.array(last_centroid)) < 50:  # Threshold for matching
                tracked_vehicles[vehicle_id] = (centroid, class_id)  # Update position
                vehicle_ids.append(vehicle_id)
                matched = True
                break

        if not matched:  # New vehicle detected
            vehicle_id = assign_vehicle_id()
            tracked_vehicles[vehicle_id] = (centroid, class_id)
            vehicle_ids.append(vehicle_id)

    # Update vehicle count
    vehicle_count = len(tracked_vehicles)

    # Draw vehicle IDs
    for vehicle_id, (centroid, class_id) in tracked_vehicles.items():
        cv2.putText(frame, f"ID: {vehicle_id}", (centroid[0] + 10, centroid[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Return total vehicle count and individual class counts
    return vehicle_count, class_counts[car_idx], class_counts[bike_idx], class_counts[truck_idx]


# Read and process video frames
frame_count = 0
# Function to update green signal time based on vehicle count
last_green_signal_time = 10  # Start with a base green signal time
update_interval = 20  # Update the green signal time every 20 frames
frame_count = 0

# Function to update green signal time based on weighted vehicle counts
def update_green_signal_time_weighted(car_count, bike_count, truck_count, base_time=10, 
                                       light_vehicle_time=0.25, car_time=0.5, heavy_vehicle_time=0.75, 
                                       max_time=90, min_time=10):
    """
    Calculate the green signal time based on the number of vehicles.
    
    Parameters:
    - car_count (int): Number of cars detected.
    - bike_count (int): Number of bikes detected.
    - truck_count (int): Number of trucks detected.
    - base_time (int): Base green signal time in seconds.
    - light_vehicle_time (int): Additional time per light vehicle (bikes).
    - car_time (float): Additional time per car.
    - heavy_vehicle_time (int): Additional time per heavy vehicle (trucks).
    - max_time (int): Maximum allowed green signal time.
    - min_time (int): Minimum allowed green signal time.
    
    Returns:
    - green_signal_time (int): Updated green signal time in seconds.
    """
    additional_time = (bike_count * light_vehicle_time + 
                       car_count * car_time + 
                       truck_count * heavy_vehicle_time)
    
    green_signal_time = base_time + additional_time
    # Round to nearest integer
    green_signal_time = int(round(green_signal_time))
    # Ensure the time does not exceed maximum or fall below minimum limits
    green_signal_time = max(min_time, min(green_signal_time, max_time))
    
    return green_signal_time

# Read and process video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Update only every `update_interval` frames
    if frame_count % update_interval == 0:
        vehicle_count, car_count, bike_count, truck_count = get_vehicle_count(frame)
        
        # Display vehicle counts on the frame
        cv2.putText(frame, f"Total Vehicles: {vehicle_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        # Calculate new green signal time based on class-weighted vehicle count
        new_green_signal_time = update_green_signal_time_weighted(car_count, bike_count, truck_count)
        
        # Ensure green signal time never decreases
        if new_green_signal_time > last_green_signal_time:
            last_green_signal_time = new_green_signal_time

    # Display the current green signal time (it won't change too frequently)
    cv2.putText(frame, f"Green Signal Time: {last_green_signal_time}s", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show the frame with vehicle counts and signal time
    cv2.imshow('Vehicle Count and Signal Time', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()