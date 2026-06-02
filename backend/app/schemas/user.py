# schemas/user.py
from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import datetime


class UserCreate(BaseModel):
    username: str
    email: EmailStr
    full_name: str = ""
    password: str


class UserOut(BaseModel):
    id: int
    username: str
    email: str
    full_name: str
    role: str
    created_at: datetime

    model_config = {"from_attributes": True}


class Token(BaseModel):
    access_token: str
    token_type: str
    user: UserOut
