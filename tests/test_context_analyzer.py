import shutil
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from src.context.context_analyzer import CodebaseScanner, ContextRelevanceAnalyzer


class TestCodebaseScanner:
    def setup_method(self):
        """Set up common test fixtures."""
        # Create a temporary directory for testing
        self.test_dir = tempfile.mkdtemp()
        self.scanner = CodebaseScanner(project_root=self.test_dir)

    def teardown_method(self):
        """Clean up after tests."""

        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_initialization(self):
        """Test CodebaseScanner initialization."""
        scanner = CodebaseScanner(project_root=self.test_dir)
        assert scanner.project_root == self.test_dir
        assert Path(scanner.project_root).exists()
        assert scanner.codebase_path == Path(self.test_dir)

    def test_scan_codebase_empty_directory(self):
        """Test scanning an empty directory."""
        result = self.scanner.scan_codebase()
        assert "file_structure" in result
        assert "raw_file_contents" in result
        assert "project_root" in result
        assert result["project_root"] == self.test_dir

    def test_scan_codebase_with_files(self):
        """Test scanning a directory with files."""
        # Create test files
        test_file = Path(self.test_dir) / "test.py"
        test_file.write_text("print('hello world')")

        docs_dir = Path(self.test_dir) / "docs"
        docs_dir.mkdir()
        doc_file = docs_dir / "readme.md"
        doc_file.write_text("# Documentation")

        result = self.scanner.scan_codebase()

        assert "file_structure" in result
        assert "raw_file_contents" in result
        assert "test.py" in result["raw_file_contents"]
        assert "docs/readme.md" in result["raw_file_contents"]
        assert result["raw_file_contents"]["test.py"] == "print('hello world')"

    def test_load_own_codebase_context(self):
        """Test loading own codebase context."""
        result = self.scanner.load_own_codebase_context()
        assert "file_structure" in result
        assert "raw_file_contents" in result
        assert result["project_root"] == self.test_dir

    def test_find_project_root(self):
        """Test finding project root."""
        root = self.scanner._find_project_root()
        assert root is not None
        assert isinstance(root, Path)

    def test_validate_project_structure(self):
        """Test validating project structure."""
        # Test with a valid directory
        project_root = Path(self.test_dir)

        # Create a minimal required file
        (project_root / "pyproject.toml").write_text("# Test")
        (project_root / "src").mkdir(exist_ok=True)
        (project_root / "src" / "__init__.py").write_text("")
        (project_root / "core.py").write_text("# Core")

        # This should run without issue (or with only warnings about missing files)
        self.scanner._validate_project_structure(project_root)

    def test_collect_raw_file_contents(self):
        """Test collecting raw file contents."""
        # Create test files
        py_file = Path(self.test_dir) / "test.py"
        py_file.write_text("print('hello')")

        md_file = Path(self.test_dir) / "readme.md"
        md_file.write_text("# Test")

        # Create an excluded directory
        excluded_dir = Path(self.test_dir) / "__pycache__"
        excluded_dir.mkdir()
        excluded_file = excluded_dir / "test.pyc"
        excluded_file.write_text("binary content")

        contents = self.scanner._collect_raw_file_contents()
        assert "test.py" in contents
        assert "readme.md" in contents
        assert "__pycache__/test.pyc" not in contents  # Should be excluded
        assert contents["test.py"] == "print('hello')"

    def test_scan_file_structure(self):
        """Test scanning file structure."""
        # Create test directory structure
        subdir = Path(self.test_dir) / "subdir"
        subdir.mkdir()
        test_file = subdir / "test.txt"
        test_file.write_text("content")

        structure = self.scanner._scan_file_structure()

        # Root directory should be in structure
        assert "." in structure
        assert "subdir" in structure["."]["subdirectories"]
        assert "subdir" in structure

        # Subdirectory should have the file
        assert "test.txt" in structure["subdir"]["files"]


class TestContextRelevanceAnalyzer:
    def setup_method(self):
        """Set up common test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.cache_dir = tempfile.mkdtemp()

        # Mock the SentenceTransformer model
        self.mock_model = Mock()
        self.mock_model.encode.return_value = [0.1, 0.2, 0.3]
        self.mock_model.get_sentence_embedding_dimension.return_value = 3
        self.mock_model.max_seq_length = 256

        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = [1, 2, 3]
        mock_tokenizer.decode.return_value = "test content"
        mock_tokenizer.model_max_length = 512
        self.mock_model.tokenizer = mock_tokenizer

    def teardown_method(self):
        """Clean up after tests."""

        shutil.rmtree(self.test_dir, ignore_errors=True)
        shutil.rmtree(self.cache_dir, ignore_errors=True)

    def test_initialization(self):
        """Test ContextRelevanceAnalyzer initialization."""
        raw_contents = {"test.py": "print('hello')"}
        analyzer = ContextRelevanceAnalyzer(
            cache_dir=self.cache_dir,
            raw_file_contents=raw_contents,
            model_name="all-MiniLM-L6-v2",
        )
        assert analyzer.cache_dir == self.cache_dir
        assert analyzer.raw_file_contents == raw_contents

    @patch("sentence_transformers.SentenceTransformer")
    def test_load_model(self, mock_sentence_transformer):
        """Test model loading."""
        # Mock the SentenceTransformer
        mock_sentence_transformer.return_value = self.mock_model
        raw_contents = {"test.py": "print('hello')"}
        analyzer = ContextRelevanceAnalyzer(
            cache_dir=self.cache_dir, raw_file_contents=raw_contents
        )
        analyzer._load_model()
        assert analyzer.model is not None

    def test_hash_codebase(self):
        """Test codebase hashing."""
        raw_contents = {"file1.py": "content1", "file2.py": "content2"}
        analyzer = ContextRelevanceAnalyzer(
            cache_dir=self.cache_dir, raw_file_contents=raw_contents
        )

        hash1 = analyzer._hash_codebase(raw_contents)
        hash2 = analyzer._hash_codebase(raw_contents)

        # Same content should produce same hash
        assert hash1 == hash2

        # Different content should produce different hash
        diff_contents = {"file1.py": "different content"}
        hash3 = analyzer._hash_codebase(diff_contents)
        assert hash1 != hash3

    @patch("sentence_transformers.SentenceTransformer")
    def test_compute_file_embeddings(self, mock_sentence_transformer):
        """Test computing file embeddings."""
        # Mock the SentenceTransformer
        mock_sentence_transformer.return_value = self.mock_model

        raw_contents = {
            "file1.py": "This is content for file 1",
            "file2.py": "This is content for file 2",
        }
        analyzer = ContextRelevanceAnalyzer(
            cache_dir=self.cache_dir,
            raw_file_contents=raw_contents,
            model_name="all-MiniLM-L6-v2",
        )

        embeddings = analyzer.compute_file_embeddings(raw_contents)

        # Should return a dictionary of embeddings
        assert isinstance(embeddings, dict)
        assert len(embeddings) == len(raw_contents)
        for file_path in raw_contents:
            assert file_path in embeddings

    def test_set_persona_router(self):
        """Test setting persona router."""
        analyzer = ContextRelevanceAnalyzer(cache_dir=self.cache_dir)
        mock_router = Mock()
        analyzer.set_persona_router(mock_router)
        assert analyzer.persona_router == mock_router

    @patch("sentence_transformers.SentenceTransformer")
    def test_find_relevant_files(self, mock_sentence_transformer):
        """Test finding relevant files."""
        # Mock the SentenceTransformer
        mock_sentence_transformer.return_value = self.mock_model
        self.mock_model.encode.return_value = [[0.1, 0.2, 0.3]]

        raw_contents = {
            "file1.py": "This is content for file 1",
            "file2.py": "This is content for file 2",
        }
        analyzer = ContextRelevanceAnalyzer(
            cache_dir=self.cache_dir,
            raw_file_contents=raw_contents,
            model_name="all-MiniLM-L6-v2",
        )

        # Compute embeddings first so there are some to search against
        analyzer.compute_file_embeddings(raw_contents)

        # Find relevant files for a prompt
        relevant_files = analyzer.find_relevant_files(
            "test prompt", max_context_tokens=1000
        )

        # Should return a list of tuples (file_path, score)
        assert isinstance(relevant_files, list)
        for file_path, score in relevant_files:
            assert isinstance(file_path, str)
            # Score can be numpy float32, numpy float64, or regular float
            assert isinstance(score, (int, float)) or str(type(score)) in [
                "<class 'numpy.float32'>",
                "<class 'numpy.float64'>",
            ]

    def test_count_tokens_robustly(self):
        """Test robust token counting."""
        # Test the fallback path to character count / 4 by making sure other methods don't exist
        mock_tokenizer = Mock()
        # Make sure the tokenizer doesn't have count_tokens method by having it raise AttributeError when accessed
        del mock_tokenizer.count_tokens  # This simulates the method not existing
        del mock_tokenizer.encode  # This simulates the method not existing

        analyzer = ContextRelevanceAnalyzer(cache_dir=self.cache_dir)
        analyzer.model = Mock()
        analyzer.model.tokenizer = mock_tokenizer

        text = "This is a test string with some content"
        tokens = analyzer._count_tokens_robustly(text)
        expected = len(text) // 4
        assert tokens == expected

    @patch("sentence_transformers.SentenceTransformer")
    def test_generate_context_summary(self, mock_sentence_transformer):
        """Test generating context summary."""
        # Mock the SentenceTransformer
        mock_sentence_transformer.return_value = self.mock_model

        raw_contents = {
            "file1.py": "This is content for file 1",
            "file2.py": "This is content for file 2",
        }
        analyzer = ContextRelevanceAnalyzer(
            cache_dir=self.cache_dir,
            raw_file_contents=raw_contents,
            model_name="all-MiniLM-L6-v2",
        )

        # Test generating summary with relevant files
        relevant_files = [("file1.py", 0.8), ("file2.py", 0.6)]
        summary = analyzer.generate_context_summary(
            relevant_files, max_tokens=1000, prompt="test prompt"
        )

        assert isinstance(summary, str)
        assert "Codebase Context for prompt" in summary
        assert "file1.py" in summary
        assert "file2.py" in summary

    def test_get_context_summary(self):
        """Test getting context summary."""
        raw_contents = {"test.py": "print('hello')"}
        analyzer = ContextRelevanceAnalyzer(
            cache_dir=self.cache_dir, raw_file_contents=raw_contents
        )

        summary = analyzer.get_context_summary()
        assert "Raw file contents available" in summary
        assert "(1 files)" in summary

        # Test with no contents
        analyzer.raw_file_contents = {}
        summary = analyzer.get_context_summary()
        assert "No raw file contents provided" in summary

    @patch("sentence_transformers.SentenceTransformer")
    def test_analyze_codebase(self, mock_sentence_transformer):
        """Test analyzing codebase."""
        # Mock the SentenceTransformer
        mock_sentence_transformer.return_value = self.mock_model

        # Create a scanner instance
        scanner = CodebaseScanner(project_root=self.test_dir)

        # Create a test file
        test_file = Path(self.test_dir) / "test.py"
        test_file.write_text("print('hello world')")

        analyzer = ContextRelevanceAnalyzer(
            cache_dir=self.cache_dir,
            raw_file_contents={},
            codebase_scanner=scanner,
            model_name="all-MiniLM-L6-v2",
        )

        # Run the analysis
        structured_context, raw_contents = analyzer.analyze_codebase()

        assert isinstance(structured_context, dict)
        assert isinstance(raw_contents, dict)
        assert "test.py" in raw_contents
        assert raw_contents["test.py"] == "print('hello world')"
