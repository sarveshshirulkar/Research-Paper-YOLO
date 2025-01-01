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

# Specify the classes we are interested in (cars, bikes, trucks, and buses as part of the truck class)
vehicle_classes = ['car', 'motorbike', 'truck']
car_idx = classes.index('car')
bike_idx = classes.index('motorbike')
truck_idx = classes.index('truck')
bus_idx = classes.index('bus')

# Load the image file
image_path = r'C:\Users\SARVESH\OneDrive\Desktop\Codezz\Python\Traffic Signal\image.jpg'  # Replace with your image path
frame = cv2.imread('type_detection.jpg')

# Check if the image is loaded correctly
if frame is None:
    raise ValueError("Could not load the image. Please check the image path.")

# Get YOLO output layer names
layer_names = net.getLayerNames()
output_layers = net.getUnconnectedOutLayers()
if len(output_layers.shape) == 1:
    output_layers = [layer_names[i - 1] for i in output_layers]
else:
    output_layers = [layer_names[i[0] - 1] for i in output_layers]

# Process image for detection
height, width = frame.shape[:2]
blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
net.setInput(blob)
detections = net.forward(output_layers)

# Initialize class counts
class_counts = {car_idx: 0, bike_idx: 0, truck_idx: 0}  # Truck will include buses

# Initialize lists to store bounding box parameters
boxes = []
confidences = []
class_ids = []

# Process detections and store information
for output in detections:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence > 0.5 and (class_id in [car_idx, bike_idx, truck_idx, bus_idx]):  # Filter for selected classes
            if class_id == bus_idx:  # Treat bus as truck
                class_id = truck_idx  # Use truck class ID

            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            width_box = int(detection[2] * width)
            height_box = int(detection[3] * height)
            x = int(center_x - width_box / 2)
            y = int(center_y - height_box / 2)

            # Append detection details
            boxes.append([x, y, width_box, height_box])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Apply Non-Maximum Suppression to filter boxes
indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)

# Draw the final bounding boxes after NMS
for i in indices.flatten():  # Use flatten to handle indices array correctly
    x, y, w, h = boxes[i]
    class_id = class_ids[i]
    
    # Define color and label
    color = (0, 255, 0)
    if class_id == car_idx:
        color = (0, 255, 255)  # Yellow for cars
    elif class_id == bike_idx:
        color = (255, 0, 0)    # Blue for bikes
    elif class_id == truck_idx:
        color = (255, 0, 255)  # Magenta for trucks (includes buses)

    # Draw bounding box and label
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 7)
    label = f"{classes[class_id]}: {confidences[i]:.2f}"
    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 6)

# Scale down the output image by 30%
scale_percent = 30
width = int(frame.shape[1] * scale_percent / 100)
height = int(frame.shape[0] * scale_percent / 100)
dim = (width, height)

# Resize the image
resized_frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

# Show the scaled-down output image with detections
cv2.imshow('Scaled Vehicle Detection', resized_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
