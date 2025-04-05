import fitz  # PyMuPDF
import requests
import logging
from collections import defaultdict
import math
from .serializers import UserSerializer
from django.conf import settings
from django.contrib.auth import get_user_model
from rest_framework import generics, status
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework_simplejwt.authentication import JWTAuthentication
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser

# Configure logging
logger = logging.getLogger(__name__)

User = get_user_model()

class RegisterView(generics.CreateAPIView):
    queryset = User.objects.all()
    serializer_class = UserSerializer
    permission_classes = [AllowAny]

def basic_analysis(resume_text, job_desc):
    """Enhanced basic keyword analysis with TF-IDF scoring"""
    # Tokenize and clean text
    resume_words = [word.lower() for word in resume_text.split() if len(word) > 2]
    job_words = [word.lower() for word in job_desc.split() if len(word) > 2]
    
    if not job_words:
        return {
            "matches": [],
            "match_percentage": 0,
            "missing_skills": [],
            "notice": "No valid keywords in job description"
        }
    
    # Calculate TF-IDF scores
    word_scores = defaultdict(float)
    total_docs = 2  # Resume and job description
    doc_counts = defaultdict(int)
    
    for word in set(resume_words + job_words):
        doc_counts[word] = (word in resume_words) + (word in job_words)
    
    for word in job_words:
        tf = job_words.count(word) / len(job_words)
        idf = math.log(total_docs / (doc_counts[word] + 1))
        word_scores[word] = tf * idf
    
    # Get matches with highest scores
    sorted_matches = sorted(
        [(word, score) for word, score in word_scores.items() if word in resume_words],
        key=lambda x: x[1],
        reverse=True
    )
    
    # Get missing skills with highest importance
    missing_skills = sorted(
        [(word, score) for word, score in word_scores.items() if word not in resume_words],
        key=lambda x: x[1],
        reverse=True
    )
    
    return {
        "matches": [word for word, score in sorted_matches[:5]],
        "match_percentage": sum(score for word, score in sorted_matches[:10]) * 10,
        "missing_skills": [word for word, score in missing_skills[:5]],
        "scoring_method": "TF-IDF"
    }

class AnalysisException(Exception):
    """Custom exception for analysis flow control"""
    def __init__(self, message, status_code, fallback_possible=False):
        self.message = message
        self.status_code = status_code
        self.fallback_possible = fallback_possible
        super().__init__(message)

class ResumeAnalysisView(APIView):
    permission_classes = [IsAuthenticated]
    authentication_classes = [JWTAuthentication]
    parser_classes = [MultiPartParser, FormParser]

    def extract_text_from_pdf(self, file_stream):
        """Safe PDF text extraction with error handling"""
        try:
            pdf_reader = fitz.open(stream=file_stream, filetype="pdf")
            resume_text = ""
            for page in pdf_reader:
                resume_text += page.get_text()[:2000]  # Limit each page
                if len(resume_text) >= 3000:  # Total limit
                    break
            return resume_text.strip()
        except Exception as e:
            logger.error(f"PDF processing error: {e}")
            raise AnalysisException(
                "Failed to process PDF file",
                status.HTTP_400_BAD_REQUEST
            )

    def analyze_with_deepseek(self, resume_text, job_description):
        """Handle DeepSeek API communication"""
        try:
            prompt = f"""Analyze this resume against the job description:
            - Provide a professional assessment in bullet points
            - Top 3 matching qualifications (with confidence scores 1-5)
            - Top 3 missing skills (with importance indicators)
            - Overall suitability percentage (0-100%)
            - 2 concrete improvement suggestions

            Resume Excerpt:
            {resume_text[:3000]}
            
            Job Description:
            {job_description[:1000]}"""

            headers = {
                "Authorization": f"Bearer {settings.DEEPSEEK_API_KEY}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "deepseek-chat",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": 800,
                "top_p": 0.9
            }
            
            response = requests.post(
                settings.DEEPSEEK_API_URL,
                headers=headers,
                json=payload,
                timeout=30
            )
            
            response.raise_for_status()
            data = response.json()
            
            if data.get("choices"):
                return data["choices"][0]["message"]["content"].strip()
                
            raise AnalysisException("No analysis generated", status.HTTP_503_SERVICE_UNAVAILABLE)
            
        except requests.exceptions.HTTPError as e:
            if response.status_code == 429:  # Rate limit
                raise AnalysisException(
                    "Analysis service overloaded - using basic analysis",
                    status.HTTP_503_SERVICE_UNAVAILABLE,
                    fallback_possible=True
                )
            raise AnalysisException(
                f"Analysis service error: {str(e)}",
                status.HTTP_503_SERVICE_UNAVAILABLE,
                fallback_possible=True
            )
        except requests.exceptions.RequestException as e:
            logger.error(f"DeepSeek API connection error: {e}")
            raise AnalysisException(
                "Analysis service unavailable - using basic analysis",
                status.HTTP_503_SERVICE_UNAVAILABLE,
                fallback_possible=True
            )

    def post(self, request, *args, **kwargs):
        file = request.FILES.get("resume")
        job_description = request.data.get("job_description", "")
        force_basic = request.query_params.get('force_basic', '').lower() == 'true'

        # Validate input
        if not file or file.size == 0:
            return Response(
                {"error": "No file provided or file is empty"},
                status=status.HTTP_400_BAD_REQUEST
            )

        if not file.name.lower().endswith('.pdf') or file.content_type != 'application/pdf':
            return Response(
                {"error": "Only valid PDF files are allowed"},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            # Read file once and reset pointer
            file_content = file.read()
            
            # Extract text
            resume_text = self.extract_text_from_pdf(file_content)
            if not resume_text:
                return Response(
                    {"error": "No readable text found in resume"},
                    status=status.HTTP_400_BAD_REQUEST
                )

            # Check for basic analysis request
            if force_basic:
                basic_result = basic_analysis(resume_text, job_description[:1000])
                return Response({
                    "analysis": "Basic keyword analysis results:",
                    **basic_result,
                    "notice": "Basic analysis by request"
                })

            try:
                # Try DeepSeek analysis first
                analysis = self.analyze_with_deepseek(resume_text, job_description)
                
                return Response({
                    "analysis": analysis,
                    "engine": "DeepSeek AI",
                    "model": "deepseek-chat"
                })
                
            except AnalysisException as e:
                # if e.fallback_possible:
                #     basic_result = basic_analysis(resume_text, job_description[:1000])
                #     return Response({
                #         "analysis": "Basic analysis results:",
                #         **basic_result,
                #         "notice": f"AI analysis failed: {e.message} - Falling back to basic analysis"
                #     })
                # return Response({"error": e.message}, status=e.status_code)
                if not resume_text:  # Critical failure (e.g., PDF parsing)
                    return Response({"error": e.message}, status=e.status_code)
        
                basic_result = basic_analysis(resume_text, job_description[:1000])
                return Response({
                    "analysis": "Basic analysis results:",
                    **basic_result,
                    "notice": f"AI analysis failed: {e.message}"
                })
            
        except AnalysisException as e:
            return Response({"error": e.message}, status=e.status_code)
            
        except Exception as e:
            logger.exception(f"Unexpected error: {e}")
            return Response(
                {
                    "error": "Internal server error",
                    "details": str(e) if settings.DEBUG else None
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )