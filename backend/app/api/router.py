from fastapi import APIRouter
from app.api.endpoints import auth, repurposing, pipeline, training, ode, ga_optimization, admin, gnn, gan
from app.api.endpoints import gnn_v2_endpoint, gan_v2_endpoint, gan_v2_train_endpoint
from app.api.endpoints import ode_sensitivity
from app.api.endpoints import gnn_xai
from app.api.endpoints import gan_xai

api_router = APIRouter()

api_router.include_router(auth.router,            prefix="/auth",        tags=["auth"])
api_router.include_router(repurposing.router,     prefix="/repurposing", tags=["tab1-repurposing"])
api_router.include_router(training.router,        prefix="/training",    tags=["tab1-training"])
api_router.include_router(pipeline.router,        prefix="/pipeline",    tags=["pipeline-session"])
api_router.include_router(ode.router,             prefix="/ode",         tags=["tab2-3-ode"])
api_router.include_router(ode_sensitivity.router, prefix="/ode",         tags=["tab3-sensitivity"])
api_router.include_router(ga_optimization.router, prefix="/ga",          tags=["tab4-ga-optimization"])
api_router.include_router(admin.router,           prefix="/admin",       tags=["admin"])
api_router.include_router(gnn.router,             prefix="/gnn",         tags=["tab5-gnn"])
api_router.include_router(gan.router,             prefix="/gan",         tags=["tab6-gan"])
api_router.include_router(gnn_v2_endpoint.router,       prefix="/gnn",   tags=["tab5-gnn-v2"])
api_router.include_router(gan_v2_endpoint.router,       prefix="/gan",   tags=["tab6-gan-v2"])
api_router.include_router(gan_v2_train_endpoint.router, prefix="/gan",   tags=["tab6-gan-v2-train"])
api_router.include_router(gnn_xai.router,               prefix="/gnn",   tags=["tab5-xai"])
api_router.include_router(gan_xai.router,               prefix="/gan",   tags=["tab6-xai"])
