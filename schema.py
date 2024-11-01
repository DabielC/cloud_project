from .database import *
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
