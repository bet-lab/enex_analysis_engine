from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers import ashpb_router

app = FastAPI(
    title="enex_analysis_engine API",
    description="API for accessing physical simulation models",
    version="1.0.0",
)

# Allow CORS for the SvelteKit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to localhost:5173 for tighter security if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ashpb_router.router, prefix="/api", tags=["ashpb"])
