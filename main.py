from .prediction import *
from .pydantic_models import *
from .database import *
from .schema import *
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from sqlalchemy.orm import joinedload

# Create a FastAPI instance
app = FastAPI()

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
    try:
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
    except:
        # Handle exceptions and return HTTP 400 error if input is invalid
        raise HTTPException(status_code=400, detail="Input the image with base64")


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
