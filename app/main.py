from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from .api.routes import router

app = FastAPI(
    title="NEET Analytics API",
    description="Student performance analysis and recommendations for NEET preparation",
    version="1.0.0"
)

app.include_router(router, prefix="/api/v1")

@app.get("/")
async def root():
    """Redirect root to API documentation"""
    return RedirectResponse(url="/docs")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
