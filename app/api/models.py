from pydantic import BaseModel
from typing import Dict, List, Optional

class QuizAnalysis(BaseModel):
    topic_accuracy: Dict[str, float]
    weak_topics: List[str]
    strong_topics: List[str]
    overall_score: float

class StudentRecommendation(BaseModel):
    recommendations: List[str]
    focus_areas: List[str]
    improvement_trends: List[str]
    suggested_difficulty: str

class AnalysisResponse(BaseModel):
    analysis: QuizAnalysis
    recommendations: StudentRecommendation 