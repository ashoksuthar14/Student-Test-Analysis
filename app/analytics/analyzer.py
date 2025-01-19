import requests
import pandas as pd
from typing import Dict, List, Tuple
from fastapi import HTTPException
import aiohttp
import plotly.express as px
import plotly.graph_objects as go
import ssl
import certifi
import google.generativeai as genai
import json
import asyncio

class QuizAnalyzer:
    def __init__(self):
        self.quiz_data_url = "https://www.jsonkeeper.com/b/LLQT"
        self.submission_data_url = "https://api.jsonserve.com/rJvd7g"
        self.historical_data_url = "https://api.jsonserve.com/XgAgFJ"
        
        # Initialize Gemini API
        genai.configure(api_key="Gemini-API")
        self.model = genai.GenerativeModel('gemini-pro')
        
        # Setup SSL context
        self.ssl_context = ssl.create_default_context(cafile=certifi.where())
        self.ssl_context.check_hostname = False
        self.ssl_context.verify_mode = ssl.CERT_NONE

    async def fetch_data(self, url: str) -> Dict:
        """Fetch data from given URL"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, ssl=self.ssl_context) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        print(f"Error status {response.status} from {url}")
                        return {}
        except Exception as e:
            print(f"Error fetching data from {url}: {str(e)}")
            return {}

    async def get_quiz_data(self) -> Tuple[Dict, List[Dict]]:
        """Analyze quiz data and return analysis and visualizations"""
        try:
            # Fetch data from all sources
            quiz_data = await self.fetch_data(self.quiz_data_url)
            if not quiz_data or 'quiz' not in quiz_data:
                raise HTTPException(status_code=500, detail="Failed to fetch quiz data")
                
            submission_data = await self.fetch_data(self.submission_data_url)
            historical_data = await self.fetch_data(self.historical_data_url)
            
            # Get topic performance
            topic_performance = self._analyze_topic_performance(quiz_data, submission_data)
            
            # Analysis results
            analysis = {
                "total_questions": len(quiz_data["quiz"]["questions"]),
                "correct_answers": submission_data.get("correct_answers", 0),
                "incorrect_answers": submission_data.get("incorrect_answers", 0),
                "accuracy": submission_data.get("accuracy", 0),
                "speed": submission_data.get("speed", 0),
                "score": submission_data.get("score", 0),
                "better_than": submission_data.get("better_than", 0),
                "topic_performance": topic_performance
            }

            # Generate visualizations
            graphs = self._generate_visualizations(analysis, topic_performance)
            
            return analysis, graphs

        except Exception as e:
            print(f"Analysis error: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    def _analyze_topic_performance(self, quiz_data: Dict, submission_data: Dict) -> Dict:
        """Analyze performance by topic"""
        try:
            topics = {}
            response_map = submission_data.get("response_map", {})
            
            for question in quiz_data.get("quiz", {}).get("questions", []):
                topic = question.get("topic", "Unknown")
                if topic not in topics:
                    topics[topic] = {
                        "correct": 0,
                        "incorrect": 0,
                        "total": 0,
                        "accuracy": 0
                    }
                
                if str(question["id"]) in response_map:
                    user_answer = response_map[str(question["id"])]
                    correct_answer = next((opt["id"] for opt in question.get("options", []) 
                                        if opt.get("is_correct")), None)
                    
                    if user_answer == correct_answer:
                        topics[topic]["correct"] += 1
                    else:
                        topics[topic]["incorrect"] += 1
                        
                topics[topic]["total"] += 1
                
                # Calculate accuracy for each topic
                total = topics[topic]["correct"] + topics[topic]["incorrect"]
                topics[topic]["accuracy"] = round((topics[topic]["correct"] / total * 100) 
                                                if total > 0 else 0, 2)
            
            return topics
        except Exception as e:
            print(f"Topic analysis error: {str(e)}")
            return {}

    def _generate_visualizations(self, analysis: Dict, topic_performance: Dict) -> List[Dict]:
        """Generate visualization data"""
        try:
            graphs = []
            
            # Overall Performance Pie Chart
            fig1 = go.Figure(data=[
                go.Pie(labels=['Correct', 'Incorrect'], 
                      values=[analysis['correct_answers'], analysis['incorrect_answers']])
            ])
            fig1.update_layout(title='Overall Performance')
            graphs.append({
                'id': 'performance_pie',
                'data': fig1.to_json()
            })

            # Topic-wise Performance Bar Chart
            if topic_performance:
                topics = list(topic_performance.keys())
                correct = [data['correct'] for data in topic_performance.values()]
                incorrect = [data['incorrect'] for data in topic_performance.values()]
                
                fig2 = go.Figure(data=[
                    go.Bar(name='Correct', x=topics, y=correct),
                    go.Bar(name='Incorrect', x=topics, y=incorrect)
                ])
                fig2.update_layout(title='Topic-wise Performance', barmode='group')
                graphs.append({
                    'id': 'topic_performance',
                    'data': fig2.to_json()
                })

            return graphs
        except Exception as e:
            print(f"Visualization error: {str(e)}")
            return []

    def analyze_quiz_data(self, quiz_data: Dict) -> Tuple[Dict, List[str]]:
        """Analyzes quiz data and returns enhanced insights."""
        try:
            quiz = quiz_data.get('quiz', {})
            questions = quiz.get('questions', [])
            
            # Basic quiz metadata
            quiz_metadata = {
                'quiz_id': quiz.get('id'),
                'title': quiz.get('title'),
                'topic': quiz.get('topic'),
                'total_questions': len(questions),
                'duration': quiz.get('duration'),
                'marks_correct': quiz.get('correct_answer_marks'),
                'marks_negative': quiz.get('negative_marks'),
                'time_per_question': f"{quiz.get('duration')/len(questions):.2f} minutes"
            }

            # Enhanced analysis
            topic_distribution = {}
            difficulty_levels = {}
            question_types = set()
            correct_options = []
            option_counts = []
            topics_by_difficulty = {}

            for question in questions:
                # Track topics
                topic = question.get('topic', 'Unknown').strip().lower()
                topic_distribution[topic] = topic_distribution.get(topic, 0) + 1
                
                # Track difficulty levels
                difficulty = question.get('difficulty_level', 'Not specified')
                difficulty_levels[difficulty] = difficulty_levels.get(difficulty, 0) + 1
                
                if difficulty not in topics_by_difficulty:
                    topics_by_difficulty[difficulty] = {}
                topics_by_difficulty[difficulty][topic] = topics_by_difficulty[difficulty].get(topic, 0) + 1

                # Track question types
                if question.get('type'):
                    question_types.add(question.get('type'))

                # Option analysis
                options = question.get('options', [])
                option_counts.append(len(options))
                
                # Track correct options
                for option in options:
                    if option.get('is_correct'):
                        correct_options.append({
                            'question_id': question.get('id'),
                            'topic': topic,
                            'correct_option': option.get('description')
                        })

            # Calculate statistics
            avg_options = sum(option_counts) / len(option_counts) if option_counts else 0
            max_score = quiz_metadata['total_questions'] * float(quiz_metadata['marks_correct'])
            min_score = -quiz_metadata['total_questions'] * float(quiz_metadata['marks_negative'])

            analysis = {
                'quiz_metadata': quiz_metadata,
                'topic_distribution': topic_distribution,
                'difficulty_distribution': difficulty_levels,
                'question_types': list(question_types) if question_types else ["No specific types found"],
                'correct_answers_sample': correct_options[:5],  # First 5 correct answers
                'statistics': {
                    'average_options_per_question': f"{avg_options:.1f}",
                    'maximum_possible_score': max_score,
                    'minimum_possible_score': min_score,
                    'time_per_question': quiz_metadata['time_per_question']
                }
            }

            # Enhanced recommendations
            recommendations = [
                f"Quiz contains {quiz_metadata['total_questions']} questions",
                f"Duration is {quiz_metadata['duration']} minutes ({quiz_metadata['time_per_question']} min/question)",
                f"Marking scheme: +{quiz_metadata['marks_correct']} for correct, -{quiz_metadata['marks_negative']} for incorrect",
                f"Maximum possible score: {max_score}",
                f"Topics covered: {', '.join(topic_distribution.keys())}",
                f"Average options per question: {avg_options:.1f}"
            ]

            return analysis, recommendations

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error analyzing quiz data: {str(e)}")

    def analyze_performance(self, student_id: str) -> Tuple[Dict, Dict]:
        # Validate student ID
        if not student_id:
            raise HTTPException(
                status_code=400,
                detail="Student ID is required"
            )

        print(f"Analyzing performance for student: {student_id}")
        
        # Use sample data instead of fetching
        quiz_data = self.sample_quiz_data
        submission_data = self.sample_submission_data
        history_data = self.sample_history_data

        # Validate data
        if not quiz_data.get("questions"):
            raise HTTPException(
                status_code=404,
                detail="No quiz data found"
            )

        if not submission_data.get("responses"):
            raise HTTPException(
                status_code=404,
                detail="No submission data found"
            )

        topic_accuracy = self._calculate_topic_accuracy(quiz_data, submission_data)
        weak_topics, strong_topics = self._identify_topic_strengths(topic_accuracy)
        overall_score = self._calculate_overall_score(quiz_data, submission_data)

        analysis = {
            "topic_accuracy": topic_accuracy,
            "weak_topics": weak_topics,
            "strong_topics": strong_topics,
            "overall_score": overall_score
        }

        recommendations = {
            "recommendations": self._generate_recommendations(weak_topics, history_data),
            "focus_areas": weak_topics,
            "improvement_trends": self._analyze_trends(history_data),
            "suggested_difficulty": self._suggest_difficulty_level(overall_score)
        }

        print(f"Analysis completed for student: {student_id}")
        return analysis, recommendations

    def _calculate_topic_accuracy(self, quiz: Dict, submission: Dict) -> Dict[str, float]:
        topic_stats = {}
        for question in quiz.get("questions", []):
            topic = question["topic"]
            if topic not in topic_stats:
                topic_stats[topic] = {"correct": 0, "total": 0}
            
            topic_stats[topic]["total"] += 1
            if submission["responses"].get(str(question["id"])) == question["correct_option"]:
                topic_stats[topic]["correct"] += 1

        return {
            topic: (stats["correct"] / stats["total"]) 
            for topic, stats in topic_stats.items()
        }

    def _identify_topic_strengths(self, accuracies: Dict[str, float]) -> Tuple[List[str], List[str]]:
        weak_topics = [topic for topic, acc in accuracies.items() if acc < 0.6]
        strong_topics = [topic for topic, acc in accuracies.items() if acc >= 0.8]
        return weak_topics, strong_topics

    def _calculate_overall_score(self, quiz: Dict, submission: Dict) -> float:
        total = len(quiz.get("questions", []))
        if total == 0:
            return 0.0
        correct = sum(
            1 for q in quiz["questions"] 
            if submission["responses"].get(str(q["id"])) == q["correct_option"]
        )
        return (correct / total) * 100

    def _generate_recommendations(self, weak_topics: List[str], history: Dict) -> List[str]:
        recommendations = []
        
        if weak_topics:
            recommendations.append(f"Focus on strengthening these topics: {', '.join(weak_topics)}")
        
        if history.get("quiz_history"):
            recent_scores = [q["score"] for q in history["quiz_history"]]
            if len(recent_scores) >= 2 and recent_scores[-1] < recent_scores[0]:
                recommendations.append("Increase your practice frequency to improve scores")
        
        if not recommendations:
            recommendations.append("Maintain your current study routine")
            
        return recommendations

    def _suggest_difficulty_level(self, overall_score: float) -> str:
        if overall_score < 40:
            return "easy"
        elif overall_score < 70:
            return "medium"
        else:
            return "hard"

    def _analyze_trends(self, history: Dict) -> List[str]:
        trends = []
        if not history.get("quiz_history"):
            return ["Not enough data to analyze trends"]

        scores = [q["score"] for q in history["quiz_history"]]
        if len(scores) >= 2:
            if scores[-1] > scores[0]:
                trends.append("Your performance is improving")
            elif scores[-1] < scores[0]:
                trends.append("Your performance needs attention")
            
        return trends 

class StudentAnalyzer:
    def __init__(self):
        self.quiz_data_url = "https://www.jsonkeeper.com/b/LLQT"
        self.submission_data_url = "https://api.jsonserve.com/rJvd7g"
        self.historical_data_url = "https://api.jsonserve.com/XgAgFJ"
        
        # Initialize Gemini API
        genai.configure(api_key="AIzaSyDNrlmsi9woPEYy1Q7O0u3fqZXgKZPk24E")
        self.model = genai.GenerativeModel('gemini-pro')
        
        # Setup SSL context
        self.ssl_context = ssl.create_default_context(cafile=certifi.where())
        self.ssl_context.check_hostname = False
        self.ssl_context.verify_mode = ssl.CERT_NONE

    async def fetch_data(self, url: str) -> Dict:
        """Fetch data from given URL with retries"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url) as response:
                        if response.status == 200:
                            data = await response.json()
                            print(f"Successfully fetched data from {url}")
                            return data
                        else:
                            print(f"Attempt {attempt + 1}: Status {response.status} from {url}")
            except Exception as e:
                print(f"Attempt {attempt + 1}: Error fetching from {url}: {str(e)}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)
        return {}

    async def get_student_analysis(self, student_id: str) -> Dict:
        """Get comprehensive student analysis"""
        try:
            # Fetch all data concurrently
            quiz_data, submission_data, historical_data = await asyncio.gather(
                self.fetch_data(self.quiz_data_url),
                self.fetch_data(self.submission_data_url),
                self.fetch_data(self.historical_data_url)
            )
            
            print("Raw Data:")
            print("Quiz Data:", quiz_data)
            print("Submission Data:", submission_data)
            print("Historical Data:", historical_data)

            # Ensure submission_data is in the correct format
            if isinstance(submission_data, str):
                submission_data = json.loads(submission_data)
            if not isinstance(submission_data, list):
                submission_data = [submission_data]

            # Extract quiz questions
            questions = quiz_data.get('quiz', {}).get('questions', [])
            if not questions:
                raise ValueError("No quiz questions found in the data")

            # Find student's submission
            student_submission = None
            for submission in submission_data:
                if isinstance(submission, dict) and str(submission.get('id')) == str(student_id):
                    student_submission = submission
                    break

            if not student_submission:
                raise HTTPException(status_code=404, detail="Student submission not found")

            # Calculate performance metrics
            response_map = student_submission.get('response_map', {})
            total_attempted = len(response_map)
            correct_answers = 0
            
            for question in questions:
                question_id = str(question.get('id'))
                if question_id in response_map:
                    correct_answer = next(
                        (opt['id'] for opt in question.get('options', []) 
                         if opt.get('is_correct')), 
                        None
                    )
                    if response_map[question_id] == correct_answer:
                        correct_answers += 1

            accuracy = (correct_answers / total_attempted * 100) if total_attempted > 0 else 0

            performance = {
                "metrics": [
                    {"name": "Overall Accuracy", "value": round(accuracy, 2), "unit": "%"},
                    {"name": "Questions Attempted", "value": total_attempted, "unit": ""},
                    {"name": "Correct Answers", "value": correct_answers, "unit": ""},
                    {"name": "Total Questions", "value": len(questions), "unit": ""}
                ]
            }

            # Topic analysis
            topics = self._analyze_topics(questions, response_map)
            
            # Historical analysis
            if isinstance(historical_data, str):
                historical_data = json.loads(historical_data)
            if not isinstance(historical_data, list):
                historical_data = [historical_data]
                
            historical = self._analyze_historical_data(historical_data, student_id)
            
            # Add quiz questions analysis
            quiz_analysis = self._analyze_quiz_questions(
                questions, 
                response_map, 
                student_submission
            )

            # Generate visualizations including historical data
            visualizations = self._generate_visualizations(
                performance, 
                topics, 
                historical,
                quiz_analysis
            )

            # Get AI recommendations
            recommendations = await self._get_ai_recommendations(
                performance, topics, historical, student_submission
            )

            return {
                "student_id": student_id,
                "performance": performance,
                "topics": topics,
                "historical": historical,
                "quiz_analysis": quiz_analysis,
                "visualizations": visualizations,
                "recommendations": recommendations
            }

        except Exception as e:
            print(f"Analysis error: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    def _analyze_topics(self, questions: List[Dict], response_map: Dict) -> List[Dict]:
        """Detailed topic-wise analysis"""
        topics = {}
        
        for question in questions:
            topic = question.get('topic', 'Unknown')
            if topic not in topics:
                topics[topic] = {
                    "name": topic,
                    "correct": 0,
                    "incorrect": 0,
                    "total": 0,
                    "accuracy": 0,
                    "questions": [],
                    "weak_areas": [],
                    "strong_areas": []
                }
            
            question_id = str(question.get('id'))
            if question_id in response_map:
                topics[topic]["total"] += 1
                correct_answer = next(
                    (opt['id'] for opt in question.get('options', []) 
                     if opt.get('is_correct')), None
                )
                
                is_correct = response_map[question_id] == correct_answer
                if is_correct:
                    topics[topic]["correct"] += 1
                    topics[topic]["strong_areas"].append(question.get('concept', 'General'))
                else:
                    topics[topic]["incorrect"] += 1
                    topics[topic]["weak_areas"].append(question.get('concept', 'General'))
                
                topics[topic]["questions"].append({
                    "id": question_id,
                    "correct": is_correct,
                    "concept": question.get('concept', 'General')
                })

        # Calculate statistics and generate recommendations
        for topic in topics.values():
            if topic["total"] > 0:
                topic["accuracy"] = round((topic["correct"] / topic["total"]) * 100, 2)
                topic["weak_areas"] = list(set(topic["weak_areas"]))
                topic["strong_areas"] = list(set(topic["strong_areas"]))
                topic["recommendation"] = self._get_topic_recommendation(
                    topic["accuracy"], 
                    topic["weak_areas"]
                )

        return list(topics.values())

    async def _get_ai_recommendations(self, performance: Dict, topics: List[Dict], 
                                    historical: Dict, submission: Dict) -> Dict:
        """Get detailed AI-powered recommendations"""
        try:
            # Prepare detailed analysis for AI
            performance_text = "\n".join(
                f"{m['name']}: {m['value']}{m['unit']}"
                for m in performance['metrics']
            )
            
            topics_text = "\n".join(
                f"Topic: {t['name']}\n"
                f"- Accuracy: {t['accuracy']}%\n"
                f"- Correct: {t['correct']}/{t['total']}\n"
                f"- Weak Areas: {', '.join(t['weak_areas'])}\n"
                f"- Strong Areas: {', '.join(t['strong_areas'])}"
                for t in topics
            )
            
            historical_text = (
                f"Historical Performance:\n"
                f"- Average Score: {historical['average_score']:.2f}\n"
                f"- Improvement Rate: {historical['improvement_rate']:.2f}\n"
                f"- Recent Scores: {', '.join(map(str, historical['trend'][:5]))}"
            )

            prompt = f"""
            As an expert educational advisor, provide a detailed analysis and personalized recommendations for this student:

            Current Performance:
            {performance_text}

            Detailed Topic Analysis:
            {topics_text}

            {historical_text}

            Please provide a comprehensive analysis including:
            1. Detailed strengths analysis - What specific skills and concepts has the student mastered?
            2. Areas needing improvement - Which specific topics and concepts need more focus?
            3. Personalized study plan - Provide a detailed weekly study schedule
            4. Advanced test-taking strategies - Based on the student's performance pattern
            5. Long-term improvement suggestions - How to maintain progress and overcome weak areas
            """

            response = self.model.generate_content(prompt)
            response_text = response.text
            
            # Parse the detailed response
            sections = response_text.split('\n\n')
            
            return {
                "strengths": sections[0] if len(sections) > 0 else "Analysis not available",
                "improvement_areas": sections[1] if len(sections) > 1 else "Analysis not available",
                "study_plan": sections[2] if len(sections) > 2 else "Analysis not available",
                "strategies": sections[3] if len(sections) > 3 else "Analysis not available",
                "long_term_goals": sections[4] if len(sections) > 4 else "Analysis not available"
            }

        except Exception as e:
            print(f"AI recommendation error: {str(e)}")
            return {
                "strengths": "Unable to generate detailed recommendations at this time",
                "improvement_areas": "System is processing the analysis",
                "study_plan": "Please try again later",
                "strategies": "Recommendations will be available soon",
                "long_term_goals": "Long-term analysis pending"
            }

    def _analyze_historical_data(self, historical_data: List[Dict], student_id: str) -> Dict:
        """Enhanced historical data analysis"""
        try:
            student_history = [
                h for h in historical_data 
                if str(h.get('id', '')) == str(student_id)
            ]
            
            if not student_history:
                return {
                    "trend": [],
                    "dates": [],
                    "scores": [],
                    "topics_progress": {},
                    "average_score": 0,
                    "improvement_rate": 0,
                    "performance_summary": "No historical data available"
                }

            # Sort by date
            history = sorted(
                student_history,
                key=lambda x: x.get('date', ''),
                reverse=True
            )

            # Extract data points
            dates = [h.get('date', '') for h in history]
            scores = [h.get('score', 0) for h in history]
            
            # Track topic-wise progress
            topics_progress = {}
            for record in history:
                for topic, score in record.get('topic_scores', {}).items():
                    if topic not in topics_progress:
                        topics_progress[topic] = []
                    topics_progress[topic].append(score)
            
            return {
                "trend": scores,
                "dates": dates,
                "scores": scores,
                "topics_progress": topics_progress,
                "average_score": sum(scores) / len(scores) if scores else 0,
                "improvement_rate": (scores[0] - scores[-1]) if len(scores) > 1 else 0,
                "performance_summary": self._get_detailed_trend_summary(scores, topics_progress)
            }
        except Exception as e:
            print(f"Historical analysis error: {str(e)}")
            return {
                "trend": [],
                "dates": [],
                "scores": [],
                "topics_progress": {},
                "average_score": 0,
                "improvement_rate": 0,
                "performance_summary": "Error analyzing historical data"
            }

    def _get_detailed_trend_summary(self, scores: List[float], topics_progress: Dict) -> str:
        """Generate detailed trend summary"""
        if not scores:
            return "No historical data available"
        
        # Overall trend analysis
        trend = "improving" if scores[0] > scores[-1] else "declining"
        consistency = "consistent" if max(scores) - min(scores) < 20 else "variable"
        
        # Topic progress analysis
        improving_topics = []
        declining_topics = []
        for topic, scores in topics_progress.items():
            if scores[0] > scores[-1]:
                improving_topics.append(topic)
            else:
                declining_topics.append(topic)
        
        summary = f"Overall performance is {trend} with {consistency} scores. "
        if improving_topics:
            summary += f"Showing improvement in: {', '.join(improving_topics)}. "
        if declining_topics:
            summary += f"Needs attention in: {', '.join(declining_topics)}."
        
        return summary

    def _analyze_quiz_questions(self, questions: List[Dict], response_map: Dict, submission: Dict) -> List[Dict]:
        """Analyze individual quiz questions with detailed feedback"""
        analyzed_questions = []
        
        for question in questions:
            question_id = str(question.get('id'))
            student_answer = response_map.get(question_id)
            
            # Find correct answer
            correct_option = next(
                (opt for opt in question.get('options', []) if opt.get('is_correct')),
                None
            )
            
            # Determine if answer was correct
            is_correct = student_answer == correct_option.get('id') if correct_option else False
            
            # Get all options with their status
            options = []
            for opt in question.get('options', []):
                option_status = {
                    'id': opt.get('id'),
                    'text': opt.get('text'),
                    'is_correct': opt.get('is_correct', False),
                    'was_selected': opt.get('id') == student_answer
                }
                options.append(option_status)
            
            # Generate question-specific feedback
            feedback = self._generate_question_feedback(
                is_correct,
                question.get('topic'),
                question.get('difficulty'),
                correct_option.get('explanation') if correct_option else None
            )
            
            analyzed_questions.append({
                'id': question_id,
                'text': question.get('text'),
                'topic': question.get('topic'),
                'difficulty': question.get('difficulty'),
                'is_correct': is_correct,
                'options': options,
                'student_answer': student_answer,
                'correct_answer': correct_option.get('id') if correct_option else None,
                'explanation': correct_option.get('explanation') if correct_option else None,
                'feedback': feedback,
                'improvement_tips': self._get_question_improvement_tips(
                    is_correct,
                    question.get('topic'),
                    question.get('difficulty')
                )
            })
        
        return analyzed_questions

    def _generate_question_feedback(self, is_correct: bool, topic: str, difficulty: str, explanation: str) -> str:
        """Generate detailed feedback for a question"""
        if is_correct:
            return f"Excellent! You've demonstrated good understanding of {topic}. " + \
                   (explanation if explanation else "Keep up the good work!")
        else:
            return f"This {difficulty} question in {topic} needs attention. " + \
                   (explanation if explanation else "Review the core concepts and try similar questions.")

    def _get_question_improvement_tips(self, is_correct: bool, topic: str, difficulty: str) -> List[str]:
        """Generate specific improvement tips for a question"""
        tips = []
        
        if not is_correct:
            tips.extend([
                f"Review the fundamental concepts of {topic}",
                f"Practice more {difficulty} level questions in {topic}",
                "Break down the question into smaller parts",
                "Look for similar patterns in practice questions"
            ])
        else:
            tips.extend([
                f"Try more challenging questions in {topic}",
                "Help explain this concept to others to reinforce learning",
                "Explore advanced applications of this concept"
            ])
        
        return tips

    def _generate_visualizations(self, performance: Dict, topics: List[Dict], 
                               historical: Dict, quiz_analysis: List[Dict]) -> List[Dict]:
        """Enhanced visualization generation"""
        graphs = []
        
        # 1. Overall Performance Gauge
        fig1 = go.Figure(data=[go.Indicator(
            mode="gauge+number",
            value=performance['metrics'][0]['value'],
            title={'text': "Overall Accuracy"},
            gauge={'axis': {'range': [0, 100]}}
        )])
        graphs.append({
            'id': 'accuracy_gauge',
            'data': fig1.to_json()
        })

        # 2. Topic Performance Bar Chart
        if topics:
            topic_names = [t['name'] for t in topics]
            accuracies = [t['accuracy'] for t in topics]
            
            fig2 = go.Figure(data=[
                go.Bar(
                    name='Accuracy',
                    x=topic_names,
                    y=accuracies,
                    text=accuracies,
                    textposition='auto',
                )
            ])
            fig2.update_layout(
                title='Topic-wise Performance',
                yaxis_title='Accuracy (%)',
                yaxis_range=[0, 100]
            )
            graphs.append({
                'id': 'topic_performance',
                'data': fig2.to_json()
            })

        # 3. Historical Performance Trend
        if historical['scores'] and historical['dates']:
            fig3 = go.Figure()
            
            # Add score line
            fig3.add_trace(go.Scatter(
                x=historical['dates'],
                y=historical['scores'],
                mode='lines+markers',
                name='Score'
            ))
            
            # Add trend line
            fig3.add_trace(go.Scatter(
                x=historical['dates'],
                y=pd.Series(historical['scores']).rolling(window=2).mean(),
                mode='lines',
                name='Trend',
                line=dict(dash='dash')
            ))
            
            fig3.update_layout(
                title='Performance History',
                xaxis_title='Date',
                yaxis_title='Score'
            )
            graphs.append({
                'id': 'trend_line',
                'data': fig3.to_json()
            })

        # 4. Topic Progress Heatmap
        if historical['topics_progress']:
            topic_names = list(historical['topics_progress'].keys())
            progress_data = [historical['topics_progress'][topic] for topic in topic_names]
            
            fig4 = go.Figure(data=go.Heatmap(
                z=progress_data,
                x=historical['dates'],
                y=topic_names,
                colorscale='RdYlGn'
            ))
            fig4.update_layout(
                title='Topic Progress Over Time',
                xaxis_title='Date',
                yaxis_title='Topic'
            )
            graphs.append({
                'id': 'topic_progress_heatmap',
                'data': fig4.to_json()
            })

        return graphs

    def _get_topic_recommendation(self, accuracy: float, weak_areas: List[str]) -> str:
        """Generate detailed topic-specific recommendation"""
        if accuracy < 50:
            return f"Critical improvement needed. Focus on: {', '.join(weak_areas)}. Start with basic concepts."
        elif accuracy < 75:
            return f"Good progress. Areas to review: {', '.join(weak_areas)}. Practice more complex problems."
        else:
            return "Excellent mastery! Focus on maintaining performance and helping others."

    def _get_topic_recommendation(self, accuracy: float, weak_areas: List[str]) -> str:
        """Generate detailed topic-specific recommendation"""
        if accuracy < 50:
            return f"Critical improvement needed. Focus on: {', '.join(weak_areas)}. Start with basic concepts."
        elif accuracy < 75:
            return f"Good progress. Areas to review: {', '.join(weak_areas)}. Practice more complex problems."
        else:
            return "Excellent mastery! Focus on maintaining performance and helping others."

    def _get_topic_recommendation(self, accuracy: float, weak_areas: List[str]) -> str:
        """Generate detailed topic-specific recommendation"""
        if accuracy < 50:
            return f"Critical improvement needed. Focus on: {', '.join(weak_areas)}. Start with basic concepts."
        elif accuracy < 75:
            return f"Good progress. Areas to review: {', '.join(weak_areas)}. Practice more complex problems."
        else:
            return "Excellent mastery! Focus on maintaining performance and helping others." 
