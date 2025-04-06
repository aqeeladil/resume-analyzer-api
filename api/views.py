import fitz  # PyMuPDF
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

from django.contrib.auth import get_user_model
from django.conf import settings

from rest_framework import generics, status
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework_simplejwt.authentication import JWTAuthentication

from .serializers import UserSerializer, AnalysisResponseSerializer

logger = logging.getLogger(__name__)
User = get_user_model()


# --- Registration View ---
class RegisterView(generics.CreateAPIView):
    queryset = User.objects.all()
    serializer_class = UserSerializer
    permission_classes = [AllowAny]


# --- TF-IDF Analysis Function ---
def basic_analysis(resume_text, job_desc):
    if not job_desc.strip():
        return {
            "matches": [],
            "match_percentage": 0,
            "missing_skills": [],
            "notice": "No valid keywords in job description"
        }

    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([resume_text[:10000], job_desc[:1000]])

    cosine_sim = (tfidf_matrix * tfidf_matrix.T).toarray()[0, 1] * 100
    feature_names = vectorizer.get_feature_names_out()

    resume_vector = tfidf_matrix[0].toarray().flatten()
    job_vector = tfidf_matrix[1].toarray().flatten()

    top_resume = sorted(zip(feature_names, resume_vector), key=lambda x: x[1], reverse=True)
    top_job = sorted(zip(feature_names, job_vector), key=lambda x: x[1], reverse=True)

    matches = [word for word, _ in top_resume[:5]]
    missing = [word for word, _ in top_job[:5] if word not in matches]

    return {
        "matches": matches,
        "match_percentage": cosine_sim,
        "missing_skills": missing,
        "scoring_method": "TF-IDF"
    }


# --- Resume Analysis API View ---
class ResumeAnalysisView(APIView):
    permission_classes = [IsAuthenticated]
    authentication_classes = [JWTAuthentication]
    parser_classes = [MultiPartParser, FormParser]

    def extract_text(self, file_stream):
        try:
            pdf = fitz.open(stream=file_stream, filetype="pdf")
            return "".join(page.get_text() for page in pdf).strip()
        except Exception as e:
            logger.error(f"PDF error: {e}")
            raise ValueError("Could not process PDF file.")

    def post(self, request, *args, **kwargs):
        file = request.FILES.get("resume")
        job_desc = request.data.get("job_description", "")

        if not file or file.size == 0:
            return Response({"error": "No resume file provided."}, status=status.HTTP_400_BAD_REQUEST)

        if not file.name.lower().endswith('.pdf') or file.content_type != 'application/pdf':
            return Response({"error": "Only PDF files are accepted."}, status=status.HTTP_400_BAD_REQUEST)

        try:
            resume_text = self.extract_text(file.read())
            if not resume_text:
                return Response({"error": "No readable text in resume."}, status=status.HTTP_400_BAD_REQUEST)

            result = basic_analysis(resume_text, job_desc)

            data = {
                "analysis": "TF-IDF Match Results",
                **result
            }
            serializer = AnalysisResponseSerializer(data=data)
            serializer.is_valid(raise_exception=True)

            return Response(serializer.data)

        except ValueError as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

        except Exception as e:
            logger.exception(f"Unexpected error: {e}")
            return Response(
                {"error": "Internal server error", "details": str(e) if settings.DEBUG else None},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
