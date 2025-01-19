from fastapi import APIRouter, HTTPException
from .models import AnalysisResponse
from ..core.state import analyzer
import json

router = APIRouter()

@router.get("/analyze-quiz")
async def analyze_quiz():
    """Analyze quiz data and provide detailed insights"""
    try:
        analysis, graphs = await analyzer.analyze_quiz_data()
        return {
            "status": "success",
            "analysis": analysis,
            "visualizations": graphs
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/quiz-data")
async def get_quiz_data():
    """Get raw quiz data from external source"""
    try:
        quiz_data = analyzer.fetch_quiz_data()
        return quiz_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/analyze/{student_id}", response_model=AnalysisResponse)
async def analyze_student(student_id: str):
    try:
        analysis, recommendations = analyzer.analyze_performance(student_id)
        return {
            "analysis": analysis,
            "recommendations": recommendations
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/available-data")
async def get_available_data():
    """Get sample data"""
    return {
        "available_student_ids": ["test123", "demo456", "sample789"],
        "message": "Use any of these student IDs with the /analyze/{student_id} endpoint"
    }

@router.get("/health")
async def health_check():
    return {"status": "healthy"} 