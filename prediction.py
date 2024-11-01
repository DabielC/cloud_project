from PIL import Image
from io import BytesIO

import base64
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torchvision.transforms.v2 as transforms
from sklearn.preprocessing import LabelEncoder
from ultralytics import YOLO

# Function to crop the face using a YOLO model
def face_crop(image):
    model = YOLO('./model/face_crop/yolov8l-face.pt')  # Load the YOLO face detection model
    
    desired_size = (190, 250)  # Desired size of the cropped face image
    padding_factor = 0.1  # Factor for padding the bounding box

    # Predict the bounding box for the face
    results = model.predict(image, show=False)
    boxes = results[0].boxes.xyxy.cpu().tolist()

    if boxes:
        box = boxes[0]

        # Calculate padding based on the bounding box dimensions
        width = box[2] - box[0]
        height = box[3] - box[1]
        pad_w = width * padding_factor
        pad_h = height * padding_factor

        # Adjust the bounding box with padding
        x1 = max(0, int(box[0] - pad_w))
        y1 = max(0, int(box[1] - pad_h))
        x2 = min(image.width, int(box[2] + pad_w))
        y2 = min(image.height, int(box[3] + pad_h))

        # Crop the object (face) from the image
        crop_obj = image.crop((x1, y1, x2, y2))

        # Resize the cropped face to the desired size
        crop_obj_resized = crop_obj.resize(desired_size)
        
        return crop_obj_resized
    else:
        return None  # Return None if no face is detected
 

# Function to convert base64 string to an image
def base64_to_image(base64_string):
    if "data:image" in base64_string:
        base64_string = base64_string.split(",")[1]  # Strip off metadata if present
    image_bytes = base64.b64decode(base64_string)  # Decode base64 string to bytes
    image = Image.open(BytesIO(image_bytes)).convert('RGB')  # Convert bytes to RGB image
    return image

# Function to apply transformations on an image for model input
def transform_image(image):
    transform = transforms.Compose([
        transforms.Resize((190, 250)),  # Resize image
        transforms.Grayscale(num_output_channels=3),  # Convert to grayscale with 3 channels
        transforms.ToTensor(),  # Convert image to tensor
        transforms.ConvertImageDtype(torch.float32),  # Convert to float32 type
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # Normalize the image
    ])
    return transform(image)

# Function to predict face type using a pre-trained MobileNet model
def predict_MobileNet(image_tensor):
    # Initialize the MobileNetV3 large model
    model = models.mobilenet_v3_large()
    num_features = model.classifier[3].in_features
    model.fc = nn.Linear(num_features, 5)  # Adjust the classifier for 5 face types
    model.load_state_dict(torch.load("./model/mobilenet_casia_web_face_augmentation/model_MobileNetV3_Greyscal_Augment.pt", map_location=torch.device('cpu')), strict=False)  # Load the model weights
    model.eval()

    # Initialize and fit the LabelEncoder for face types
    label_encoder = LabelEncoder()
    label_encoder.fit(['Oblong', 'Round', 'Oval', 'Heart', 'Square'])  # Define class labels
    model.eval()
    
    with torch.no_grad():  # Disable gradient calculation for inference
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
        outputs = model(image_tensor)  # Get model predictions
        
        # Calculate softmax probabilities
        confidences = F.softmax(outputs, dim=1)
        predicted_class_index = torch.argmax(confidences, dim=1).item()  # Get the class with highest probability
        predicted_class_label = label_encoder.inverse_transform([predicted_class_index])[0]  # Get the corresponding class label
        confidence_scores = confidences.squeeze().tolist()  # Convert to list
        confidence_mapping = {label: score for label, score in zip(label_encoder.classes_, confidence_scores)}  # Map classes to confidence scores
        
    return {"class" : predicted_class_label, "score" : confidence_mapping}

# Function to predict face type using a YOLO model
def predict_YOLO(image):
    model = YOLO('./model/yolov8_imagenet/trained_yolov8x-cls_2.pt')  # Load the YOLO classification model
    results = model(image)  # Predict the face type using YOLO

    predicted_class_index = results[0].probs.top1  # Get the class with the highest probability
    predicted_class_name = results[0].names[predicted_class_index]  # Get the corresponding class name

    dict = {}
    for i, prob in enumerate(results[0].probs.data.tolist()):
        dict[results[0].names[i]] = prob  # Map the class names to their probabilities
    return {"class" : predicted_class_name, "score" : dict}

# Function to combine predictions from MobileNet and YOLO models
def vote(mobile_pred, yolo_pred):
    mobile_conf = mobile_pred["score"]  # Get MobileNet confidence scores
    yolo_conf = yolo_pred["score"]  # Get YOLO confidence scores
    average_score = {
        'Heart': (yolo_conf['Heart'] + mobile_conf['Heart']) / 2,
        'Oblong': (yolo_conf['Oblong'] + mobile_conf['Oblong']) / 2,
        'Oval': (yolo_conf['Oval'] + mobile_conf['Oval']) / 2,
        'Round': (yolo_conf['Round'] + mobile_conf['Round']) / 2,
        'Square': (yolo_conf['Square'] + mobile_conf['Square']) / 2
    }  # Calculate the average of both models' confidence scores for each class

    voted_class = max(average_score, key=average_score.get)  # Get the class with the highest average score
    
    return {"class" : voted_class, "score": average_score}  # Return the voted class and the average scores
