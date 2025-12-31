"""
Unit tests for SearchResults dataclass.

Tests evaluate:
1. Factory methods (from_chroma, empty)
2. is_empty method
3. Attribute storage
"""

import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from vector_store import SearchResults


class TestSearchResultsCreation:
    """Tests for SearchResults creation."""

    def test_create_with_documents_and_metadata(self):
        """Verify basic creation with documents and metadata."""
        results = SearchResults(
            documents=["doc1", "doc2"],
            metadata=[{"key": "val1"}, {"key": "val2"}],
            distances=[0.3, 0.5],
        )

        assert len(results.documents) == 2
        assert len(results.metadata) == 2
        assert len(results.distances) == 2

    def test_create_with_error(self):
        """Verify creation with error message."""
        results = SearchResults(
            documents=[], metadata=[], distances=[], error="Something went wrong"
        )

        assert results.error == "Something went wrong"

    def test_error_defaults_to_none(self):
        """Verify error is None by default."""
        results = SearchResults(documents=["doc"], metadata=[{}], distances=[0.1])

        assert results.error is None


class TestSearchResultsFromChroma:
    """Tests for from_chroma factory method."""

    def test_from_chroma_extracts_documents(self):
        """Verify documents are extracted from ChromaDB format."""
        chroma_results = {
            "documents": [["doc1", "doc2"]],
            "metadatas": [[{}, {}]],
            "distances": [[0.3, 0.5]],
        }

        results = SearchResults.from_chroma(chroma_results)

        assert results.documents == ["doc1", "doc2"]

    def test_from_chroma_extracts_metadata(self):
        """Verify metadata is extracted from ChromaDB format."""
        chroma_results = {
            "documents": [["doc"]],
            "metadatas": [[{"course_title": "ML", "lesson_number": 1}]],
            "distances": [[0.3]],
        }

        results = SearchResults.from_chroma(chroma_results)

        assert results.metadata[0]["course_title"] == "ML"
        assert results.metadata[0]["lesson_number"] == 1

    def test_from_chroma_extracts_distances(self):
        """Verify distances are extracted from ChromaDB format."""
        chroma_results = {
            "documents": [["doc1", "doc2"]],
            "metadatas": [[{}, {}]],
            "distances": [[0.25, 0.75]],
        }

        results = SearchResults.from_chroma(chroma_results)

        assert results.distances == [0.25, 0.75]

    def test_from_chroma_handles_empty_results(self):
        """Verify empty ChromaDB results are handled."""
        chroma_results = {"documents": [], "metadatas": [], "distances": []}

        results = SearchResults.from_chroma(chroma_results)

        assert results.documents == []
        assert results.metadata == []
        assert results.distances == []

    def test_from_chroma_handles_none_in_nested_arrays(self):
        """Verify handling of nested empty arrays."""
        chroma_results = {"documents": [[]], "metadatas": [[]], "distances": [[]]}

        results = SearchResults.from_chroma(chroma_results)

        assert results.documents == []
        assert results.metadata == []
        assert results.distances == []


class TestSearchResultsEmpty:
    """Tests for empty factory method."""

    def test_empty_creates_error_result(self):
        """Verify empty factory creates result with error."""
        results = SearchResults.empty("No results found")

        assert results.error == "No results found"
        assert results.documents == []
        assert results.metadata == []
        assert results.distances == []

    def test_empty_with_different_error_messages(self):
        """Verify different error messages are stored."""
        results1 = SearchResults.empty("Error 1")
        results2 = SearchResults.empty("Error 2")

        assert results1.error == "Error 1"
        assert results2.error == "Error 2"


class TestSearchResultsIsEmpty:
    """Tests for is_empty method."""

    def test_is_empty_true_when_no_documents(self):
        """Verify is_empty returns True for empty documents."""
        results = SearchResults(documents=[], metadata=[], distances=[])

        assert results.is_empty() is True

    def test_is_empty_false_when_documents_present(self):
        """Verify is_empty returns False when documents exist."""
        results = SearchResults(
            documents=["some content"], metadata=[{}], distances=[0.5]
        )

        assert results.is_empty() is False

    def test_is_empty_with_error_and_no_documents(self):
        """Verify is_empty is True even with error set."""
        results = SearchResults.empty("An error occurred")

        assert results.is_empty() is True

    def test_is_empty_single_document(self):
        """Verify is_empty is False with single document."""
        results = SearchResults(
            documents=["single doc"], metadata=[{"key": "value"}], distances=[0.1]
        )

        assert results.is_empty() is False


class TestSearchResultsAttributes:
    """Tests for attribute access."""

    def test_documents_accessible(self):
        """Verify documents attribute is accessible."""
        results = SearchResults(
            documents=["content here"], metadata=[{}], distances=[0.0]
        )

        assert results.documents[0] == "content here"

    def test_metadata_accessible(self):
        """Verify metadata attribute is accessible."""
        results = SearchResults(
            documents=["doc"], metadata=[{"title": "Test Course"}], distances=[0.0]
        )

        assert results.metadata[0]["title"] == "Test Course"

    def test_distances_accessible(self):
        """Verify distances attribute is accessible."""
        results = SearchResults(documents=["doc"], metadata=[{}], distances=[0.42])

        assert results.distances[0] == 0.42

    def test_all_lists_same_length(self):
        """Verify all lists maintain same length relationship."""
        docs = ["a", "b", "c"]
        meta = [{"i": 0}, {"i": 1}, {"i": 2}]
        dist = [0.1, 0.2, 0.3]

        results = SearchResults(documents=docs, metadata=meta, distances=dist)

        assert len(results.documents) == len(results.metadata) == len(results.distances)
