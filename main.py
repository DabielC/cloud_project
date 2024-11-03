# from .prediction import *
# from .pydantic_models import *
# from .database import *
# from .schema import *


from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import os
from dotenv import load_dotenv
load_dotenv()

# Define the database URL for the AWS RDS MySQL database
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
SQLALCHEMY_DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Create an engine that connects to the RDS MySQL database
engine = create_engine(SQLALCHEMY_DATABASE_URL)

# Create a configured "Session" class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create a Session object that can be used to interact with the database
session = Session(engine)

# Base class for declarative class definitions
Base = declarative_base()

# Dependency to get a database session
def get_db():
    db = SessionLocal()  # Instantiate a new database session
    try:
        yield db  # Yield the session to be used in the calling function
    finally:
        db.close()  # Ensure that the session is closed after use

# Function to check if the database is connected
def is_db_connected():
    try:
        # Attempt to connect to the database
        with engine.connect() as connection:
            return "Database connection successful"
    except Exception as e:
        return f"Database connection failed: {e}"

from sqlalchemy import *
from sqlalchemy.orm import relationship
from sqlalchemy import ForeignKey

# Model for face_glasses table, which links face types to glasses types
class face_glasses(Base):
    __tablename__ = 'face_glasses'
    id = Column(Integer, primary_key=True)  # Primary key
    face_type = Column(String, ForeignKey('face_shape.face_type'))  # Foreign key referencing face_shape table
    glasses_type = Column(String, ForeignKey('glasses_class.glasses_type'))  # Foreign key referencing glasses_class table
    suitability = Column(String)  # Suitability score/description for this face-glasses pairing

    # Relationships
    face_shape = relationship("face_shape", back_populates="face_glasses")  # Relationship to face_shape
    glasses_class = relationship("glasses_class", back_populates="face_glasses")  # Relationship to glasses_class


# Model for face_shape table, which stores different face types
class face_shape(Base):
    __tablename__ = 'face_shape'
    face_type = Column(String, primary_key=True)  # Primary key, face type

    # Relationship
    face_glasses = relationship("face_glasses", back_populates="face_shape")  # One-to-many relationship with face_glasses


# Model for glasses_class table, which stores different types of glasses
class glasses_class(Base):
    __tablename__ = 'glasses_class'
    glasses_type = Column(String, primary_key=True)  # Primary key, glasses type

    # Relationships
    face_glasses = relationship("face_glasses", back_populates="glasses_class")  # One-to-many relationship with face_glasses
    glasses_product = relationship("glasses_product", back_populates="glasses_class")  # One-to-many relationship with glasses_product


# Model for glasses_product table, which stores glasses product details
class glasses_product(Base):
    __tablename__ = 'glasses_product'
    id = Column(Integer, primary_key=True)  # Primary key
    glasses_type = Column(String, ForeignKey('glasses_class.glasses_type'))  # Foreign key referencing glasses_class table
    glasses_img = Column(String)  # Image of the glasses

    # Relationships
    glasses_class = relationship("glasses_class", back_populates="glasses_product")  # Relationship to glasses_class


# Model for user_reflection table, which stores user interactions and scores
class user_reflection(Base):
    __tablename__ = 'user_reflection'
    id = Column(Integer, primary_key=True)  # Primary key
    like = Column(Integer)  # Number of likes
    comment = Column(String)  # User comment
    img = Column(String)  # Image associated with the reflection
    create_at = Column(String)  # Timestamp of creation
    mobilenet_score = Column(String)  # Score from MobileNet model
    yolov8_score = Column(String)  # Score from YOLOv8 model
    vote_score = Column(String)  # Final vote score from combined models

from pydantic import BaseModel
from typing import Optional

# Request model for receiving image data
class get_info(BaseModel):
    image: str  # Base64 encoded image string

# Request model for updating the 'like' field
class update_like(BaseModel):
    id: int  # ID of the item to update
    like: Optional[int] = None  # Optional like field, can be None if not provided

# Request model for updating the 'comment' field
class update_comment(BaseModel):
    id: int  # ID of the item to update
    comment : str  # New comment for the item

# Request model for creating a new item, inherits from get_info
class ItemCreate(get_info):
    image: str  # Base64 encoded image string for item creation



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
    model = YOLO('./model/face_crop/yolov8l-face (1).pt')  # Load the YOLO face detection model
    
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
    model = YOLO('./model/yolov8_imagenet/trained_yolov8x-cls_2 (1).pt')  # Load the YOLO classification model
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




from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from sqlalchemy.orm import joinedload
import ssl
import uvicorn

# Create a FastAPI instance
app = FastAPI()

ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
ssl_context.load_cert_chain('../cert.pem', keyfile='../key.pem')

# Add CORS middleware to allow requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Route for image prediction
@app.post("/predict")
async def predicted(info: ItemCreate, db: Session = Depends(get_db)):
    # Detect face in the provided image
    face_detect = face_crop(base64_to_image(info.image))
    if face_detect is not None:
        # If face is detected, transform and predict using models
        image_tensor = transform_image(face_detect)
        predicted_mobile = predict_MobileNet(image_tensor)
        predicted_yolo = predict_YOLO(face_detect)
        vote_score = vote(predicted_mobile, predicted_yolo)
        
        # Save user reflection (prediction data) to the database
        db_item = user_reflection(
            img=info.image,
            create_at=str(datetime.now()),
            mobilenet_score=str(predicted_mobile),
            yolov8_score=str(predicted_yolo),
            vote_score=str(vote_score)
        )
        
        # Query the database for matching glasses products based on face type
        db_select = (
            db.query(face_glasses, glasses_product)
            .join(glasses_product, face_glasses.glasses_type == glasses_product.glasses_type)
            .filter(face_glasses.face_type == vote_score["class"])
            .options(joinedload(face_glasses.glasses_class))
            .all()
        )

        # Add new reflection to the database
        db.add(db_item)
        db.commit()
        db.refresh(db_item)

        # Prepare the glasses product data to return
        products = []
        for fg, gp in db_select:
            products.append({
                "glasses_type": fg.glasses_type,
                "suitability": fg.suitability,
                "glasses_img": gp.glasses_img
            })

        # Return prediction scores and suitable glasses products
        return {
                "id": db_item.id,
                "mscore": predicted_mobile["score"],
                "yscore": predicted_yolo["score"],
                "vscore": vote_score,
                "dt": db_item.create_at,
                "products": products
            }
    
    else:
        # If no face detected, save the reflection and return the message
        db_item = user_reflection(
            img=info.image,
            create_at=str(datetime.now())
        )
        db.add(db_item)
        db.commit()
        db.refresh(db_item)
        return "No face detected"


# Route to get image by item ID from the database
@app.get("/get_image/{item_id}")
def test_db(item_id : int, db: Session = Depends(get_db)):
    db_item = db.query(user_reflection).filter(user_reflection.id == item_id).first()
    return db_item.img

# Route to update 'like' field of a user reflection
@app.put("/update_reflection_like")
async def update_user_reflection(item: update_like, db: Session = Depends(get_db)):
    db_item = db.query(user_reflection).filter(user_reflection.id == item.id).first()
    if db_item:
        # Update fields based on the provided data
        for key, value in item.model_dump().items():
            setattr(db_item, key, value)
        db.commit()  # Commit the update
        db.refresh(db_item)  # Refresh the instance to return updated data
        return 'Update like Success!'
    else:
        # Raise 404 error if item is not found
        raise HTTPException(status_code=404, detail="Item not found")
    
# Route to update 'comment' field of a user reflection
@app.put("/update_reflection_comment")
async def update_user_reflection(item: update_comment, db: Session = Depends(get_db)):
    db_item = db.query(user_reflection).filter(user_reflection.id == item.id).first()
    if db_item:
        # Update fields based on the provided data
        for key, value in item.model_dump().items():
            setattr(db_item, key, value)
        db.commit()  # Commit the update
        db.refresh(db_item)  # Refresh the instance to return updated data
        return 'Update comment Success!'
    else:
        # Raise 404 error if item is not found
        raise HTTPException(status_code=404, detail="Item not found")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        ssl_keyfile="../key.pem",
        ssl_certfile="../cert.pem"
    )