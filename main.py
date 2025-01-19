from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from app.core.state import analyzer

app = FastAPI()
templates = Jinja2Templates(directory="app/templates")

@app.get("/student/{student_id}", response_class=HTMLResponse)
async def student_dashboard(request: Request, student_id: str):
    try:
        student_id = student_id.replace("{", "").replace("}", "")
        analysis = await analyzer.get_student_analysis(student_id)
        return templates.TemplateResponse(
            "student/dashboard.html",
            {
                "request": request,
                "analysis": analysis,
                "student_id": student_id
            }
        )
    except Exception as e:
        print(f"Dashboard error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Visit /student/{student_id} for student analysis"}