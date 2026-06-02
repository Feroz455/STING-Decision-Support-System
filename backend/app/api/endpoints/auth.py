from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.core.database import get_db, User
from app.core.security import verify_password, create_access_token, hash_password
from app.schemas.user import UserCreate, UserOut, Token

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)


@router.post("/token", response_model=Token)
@limiter.limit("10/minute")
async def login(
    request: Request,
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(User).where(User.username == form_data.username))
    user = result.scalar_one_or_none()

    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Kullanıcı adı veya şifre hatalı",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not user.is_active:
        raise HTTPException(status_code=403, detail="Hesap devre dışı")

    token = create_access_token({"sub": user.username, "role": user.role, "uid": user.id})
    return {"access_token": token, "token_type": "bearer", "user": UserOut.model_validate(user)}


@router.post("/register", response_model=UserOut, status_code=201)
async def register(payload: UserCreate, db: AsyncSession = Depends(get_db)):
    # Check duplicate
    existing = await db.execute(select(User).where(User.username == payload.username))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Bu kullanıcı adı zaten alınmış")

    user = User(
        username=payload.username,
        email=payload.email,
        full_name=payload.full_name,
        hashed_password=hash_password(payload.password),
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return UserOut.model_validate(user)


@router.get("/me", response_model=UserOut)
async def me(db: AsyncSession = Depends(get_db), token: str = Depends(__import__("app.core.security", fromlist=["oauth2_scheme"]).oauth2_scheme)):
    from app.core.security import decode_token
    payload = decode_token(token)
    result = await db.execute(select(User).where(User.username == payload["sub"]))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="Kullanıcı bulunamadı")
    return UserOut.model_validate(user)
