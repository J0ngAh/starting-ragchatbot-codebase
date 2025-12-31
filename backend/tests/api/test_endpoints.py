"""
API endpoint tests for the RAG chatbot.

This module creates a test-specific FastAPI app to avoid import issues
with static file mounting in the main app.py.
"""
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from unittest.mock import MagicMock, patch


# =============================================================================
# Test App Definition (mirrors app.py without static file mounting)
# =============================================================================

class QueryRequest(BaseModel):
    """Request model for course queries"""
    query: str
    session_id: Optional[str] = None


class QueryResponse(BaseModel):
    """Response model for course queries"""
    answer: str
    sources: List[Dict[str, Any]]
    session_id: str


class CourseStats(BaseModel):
    """Response model for course statistics"""
    total_courses: int
    course_titles: List[str]


def create_test_app(mock_rag_system: MagicMock) -> FastAPI:
    """Create a test FastAPI app with mocked RAG system."""
    app = FastAPI(title="Course Materials RAG System - Test")

    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        """Process a query and return response with sources"""
        try:
            session_id = request.session_id
            if not session_id:
                session_id = mock_rag_system.session_manager.create_session()

            answer, sources = mock_rag_system.query(request.query, session_id)

            return QueryResponse(
                answer=answer,
                sources=sources,
                session_id=session_id
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        """Get course analytics and statistics"""
        try:
            analytics = mock_rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/")
    async def root():
        """Health check endpoint"""
        return {"status": "ok", "message": "RAG System API"}

    return app


# =============================================================================
# Test Client Fixtures
# =============================================================================

@pytest.fixture
def test_client(mock_rag_system):
    """Create test client with mocked RAG system."""
    app = create_test_app(mock_rag_system)
    return TestClient(app)


@pytest.fixture
def test_client_with_errors(mock_rag_system_error):
    """Create test client with RAG system that raises errors."""
    app = create_test_app(mock_rag_system_error)
    return TestClient(app)


# =============================================================================
# Query Endpoint Tests
# =============================================================================

@pytest.mark.api
class TestQueryEndpoint:
    """Tests for POST /api/query endpoint."""

    def test_query_success_without_session(self, test_client, sample_query_request):
        """Test successful query without existing session."""
        response = test_client.post("/api/query", json=sample_query_request)

        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
        assert data["session_id"] == "test_session_123"
        assert "test response" in data["answer"].lower()

    def test_query_success_with_session(self, test_client, sample_query_request_with_session):
        """Test successful query with existing session."""
        response = test_client.post("/api/query", json=sample_query_request_with_session)

        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "existing_session_456"

    def test_query_returns_sources(self, test_client, sample_query_request):
        """Test that query returns source citations."""
        response = test_client.post("/api/query", json=sample_query_request)

        assert response.status_code == 200
        data = response.json()
        assert len(data["sources"]) > 0
        source = data["sources"][0]
        assert "title" in source
        assert "url" in source

    def test_query_invalid_request_missing_query(self, test_client):
        """Test query with missing required field."""
        response = test_client.post("/api/query", json={})

        assert response.status_code == 422  # Validation error

    def test_query_invalid_request_wrong_type(self, test_client):
        """Test query with wrong field type."""
        response = test_client.post("/api/query", json={"query": 123})

        assert response.status_code == 422

    def test_query_empty_string(self, test_client):
        """Test query with empty string (should still process)."""
        response = test_client.post("/api/query", json={"query": ""})

        # Empty string is valid according to the model, RAG system handles it
        assert response.status_code == 200

    def test_query_internal_error(self, test_client_with_errors, sample_query_request):
        """Test query endpoint handles internal errors gracefully."""
        response = test_client_with_errors.post("/api/query", json=sample_query_request)

        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        assert "Internal RAG system error" in data["detail"]


# =============================================================================
# Courses Endpoint Tests
# =============================================================================

@pytest.mark.api
class TestCoursesEndpoint:
    """Tests for GET /api/courses endpoint."""

    def test_get_courses_success(self, test_client):
        """Test successful retrieval of course stats."""
        response = test_client.get("/api/courses")

        assert response.status_code == 200
        data = response.json()
        assert "total_courses" in data
        assert "course_titles" in data
        assert data["total_courses"] == 3
        assert len(data["course_titles"]) == 3

    def test_get_courses_returns_list(self, test_client):
        """Test that course_titles is a list."""
        response = test_client.get("/api/courses")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data["course_titles"], list)
        assert "Course A" in data["course_titles"]

    def test_get_courses_internal_error(self, test_client_with_errors):
        """Test courses endpoint handles internal errors gracefully."""
        response = test_client_with_errors.get("/api/courses")

        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        assert "Failed to get analytics" in data["detail"]


# =============================================================================
# Root Endpoint Tests
# =============================================================================

@pytest.mark.api
class TestRootEndpoint:
    """Tests for GET / endpoint."""

    def test_root_health_check(self, test_client):
        """Test root endpoint returns health status."""
        response = test_client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"


# =============================================================================
# Request/Response Model Tests
# =============================================================================

@pytest.mark.api
class TestRequestResponseModels:
    """Tests for request/response Pydantic models."""

    def test_query_request_optional_session(self, test_client):
        """Test that session_id is truly optional."""
        # Without session_id
        response1 = test_client.post("/api/query", json={"query": "test"})
        assert response1.status_code == 200

        # With explicit None
        response2 = test_client.post("/api/query", json={"query": "test", "session_id": None})
        assert response2.status_code == 200

        # With session_id
        response3 = test_client.post("/api/query", json={"query": "test", "session_id": "abc123"})
        assert response3.status_code == 200

    def test_query_response_structure(self, test_client, sample_query_request):
        """Test that query response has correct structure."""
        response = test_client.post("/api/query", json=sample_query_request)

        assert response.status_code == 200
        data = response.json()

        # Verify all required fields present
        assert set(data.keys()) == {"answer", "sources", "session_id"}

        # Verify types
        assert isinstance(data["answer"], str)
        assert isinstance(data["sources"], list)
        assert isinstance(data["session_id"], str)

    def test_course_stats_response_structure(self, test_client):
        """Test that course stats response has correct structure."""
        response = test_client.get("/api/courses")

        assert response.status_code == 200
        data = response.json()

        # Verify all required fields present
        assert set(data.keys()) == {"total_courses", "course_titles"}

        # Verify types
        assert isinstance(data["total_courses"], int)
        assert isinstance(data["course_titles"], list)


# =============================================================================
# Edge Cases and Boundary Tests
# =============================================================================

@pytest.mark.api
class TestEdgeCases:
    """Edge case tests for API endpoints."""

    def test_query_with_special_characters(self, test_client):
        """Test query with special characters."""
        special_query = "What about <script>alert('xss')</script>?"
        response = test_client.post("/api/query", json={"query": special_query})

        # Should process without issues (RAG system handles sanitization)
        assert response.status_code == 200

    def test_query_with_unicode(self, test_client):
        """Test query with unicode characters."""
        unicode_query = "What about machine learning?"
        response = test_client.post("/api/query", json={"query": unicode_query})

        assert response.status_code == 200

    def test_query_with_very_long_input(self, test_client):
        """Test query with very long input."""
        long_query = "What is " + "a" * 10000 + "?"
        response = test_client.post("/api/query", json={"query": long_query})

        # Should process (may be truncated by RAG system)
        assert response.status_code == 200

    def test_content_type_json_required(self, test_client):
        """Test that JSON content type is required."""
        response = test_client.post(
            "/api/query",
            content="query=test",
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )

        assert response.status_code == 422
