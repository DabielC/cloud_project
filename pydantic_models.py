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
