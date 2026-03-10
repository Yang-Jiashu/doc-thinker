"""
Configuration classes for DocThinker

Contains configuration dataclasses with environment variable support
"""
#封装了所有的环境变量以及参数的配置及默认值，和工作目录。
from dataclasses import dataclass, field
from typing import List
from graphcore.coregraph.utils import get_env_value


@dataclass
class DocThinkerConfig:
    """Configuration class for DocThinker with environment variable support"""

    # Directory Configuration
    # ---
    working_dir: str = field(default=get_env_value("WORKING_DIR", "./doc_storage", str))
    """Directory where RAG storage and cache files are stored."""

    # Parser Configuration
    # ---
    parse_method: str = field(default=get_env_value("PARSE_METHOD", "auto", str))
    """Default parsing method for document parsing: 'auto', 'ocr', or 'txt'."""

    parser_output_dir: str = field(default=get_env_value("OUTPUT_DIR", "./output", str))
    """Default output directory for parsed content."""

    parser: str = field(default=get_env_value("PARSER", "mineru", str))
    """Parser selection: 'mineru' or 'docling'."""

    display_content_stats: bool = field(
        default=get_env_value("DISPLAY_CONTENT_STATS", True, bool)
    )
    """Whether to display content statistics during parsing."""

    # Multimodal Processing Configuration
    # ---
    enable_image_processing: bool = field(
        default=get_env_value("ENABLE_IMAGE_PROCESSING", True, bool)
    )
    """Enable image content processing."""

    enable_table_processing: bool = field(
        default=get_env_value("ENABLE_TABLE_PROCESSING", True, bool)
    )
    """Enable table content processing."""

    enable_equation_processing: bool = field(
        default=get_env_value("ENABLE_EQUATION_PROCESSING", True, bool)
    )
    """Enable equation content processing."""

    # Batch Processing Configuration
    # ---
    max_concurrent_files: int = field(
        default=get_env_value("MAX_CONCURRENT_FILES", 1, int)
    )
    """Maximum number of files to process concurrently."""

    supported_file_extensions: List[str] = field(
        default_factory=lambda: get_env_value(
            "SUPPORTED_FILE_EXTENSIONS",
            ".pdf,.jpg,.jpeg,.png,.bmp,.tiff,.tif,.gif,.webp,.doc,.docx,.ppt,.pptx,.xls,.xlsx,.txt,.md",
            str,
        ).split(",")
    )
    """List of supported file extensions for batch processing."""

    recursive_folder_processing: bool = field(
        default=get_env_value("RECURSIVE_FOLDER_PROCESSING", True, bool)
    )
    """Whether to recursively process subfolders in batch mode."""

    # Context Extraction Configuration
    # ---
    context_window: int = field(default=get_env_value("CONTEXT_WINDOW", 1, int))
    """Number of pages/chunks to include before and after current item for context."""

    context_mode: str = field(default=get_env_value("CONTEXT_MODE", "page", str))
    """Context extraction mode: 'page' for page-based, 'chunk' for chunk-based."""

    max_context_tokens: int = field(
        default=get_env_value("MAX_CONTEXT_TOKENS", 3200, int)
    )
    """Maximum number of tokens in extracted context."""

    include_headers: bool = field(default=get_env_value("INCLUDE_HEADERS", True, bool))
    """Whether to include document headers and titles in context."""

    include_captions: bool = field(
        default=get_env_value("INCLUDE_CAPTIONS", True, bool)
    )
    """Whether to include image/table captions in context."""

    context_filter_content_types: List[str] = field(
        default_factory=lambda: get_env_value(
            "CONTEXT_FILTER_CONTENT_TYPES", "text", str
        ).split(",")
    )
    """Content types to include in context extraction (e.g., 'text', 'image', 'table')."""

    content_format: str = field(default=get_env_value("CONTENT_FORMAT", "minerU", str))
    """Default content format for context extraction when processing documents."""

    # Auto-thinking Configuration
    # ---
    enable_auto_thinking: bool = field(
        default=get_env_value("ENABLE_AUTO_THINKING", False, bool)
    )
    """Enable hybrid auto-thinking orchestration."""

    # Relation & Entity Extraction Configuration
    # ---
    relation_extraction_mode: str = field(default=get_env_value("RELATION_EXTRACTION_MODE", "graphcore", str))
    """Strategy for extracting relations/entities: 'graphcore' or 'hypergraph'."""

    graph_construction_mode: str = field(default=get_env_value("GRAPH_CONSTRUCTION_MODE", "llm", str))
    """Mode for graph construction: 'llm' (extract entities and relations via LLM) or 'linear' (extract entities via NER)."""

    spacy_model: str = field(default=get_env_value("SPACY_MODEL", "en_core_web_sm", str))
    """Spacy model to use for entity extraction in 'linear' mode."""

    hyper_prompt_language: str = field(
        default=get_env_value("HYPER_PROMPT_LANGUAGE", "en", str)
    )
    """Default language hint for HyperGraph prompt based extraction."""

    hyper_prompt_example_number: int = field(
        default=get_env_value("HYPER_PROMPT_EXAMPLE_NUMBER", 4, int)
    )
    """Number of in-context examples to include when using HyperGraph prompts."""

    hyper_prompt_max_gleaning: int = field(
        default=get_env_value("HYPER_PROMPT_MAX_GLEANING", 1, int)
    )
    """Maximum extra gleaning rounds when HyperGraph prompt continues generation."""

    hyper_prompt_entity_types: List[str] = field(
        default_factory=lambda: get_env_value(
            "HYPER_PROMPT_ENTITY_TYPES",
            "person,organization,location,event",
            str,
        ).split(",")
    )
    """Entity type hints (comma separated)."""

    auto_thinking_sync_mode: str = field(
        default=get_env_value("AUTO_THINKING_SYNC_MODE", "lazy", str)
    )
    """Synchronisation mode for HyperGraphRAG chunks: 'lazy' or 'eager'."""

    hyper_chunk_token_size: int = field(
        default=get_env_value("HYPER_CHUNK_TOKEN_SIZE", 1200, int)
    )
    """Token window used when slicing text for HyperGraphRAG."""

    hyper_chunk_overlap: int = field(
        default=get_env_value("HYPER_CHUNK_OVERLAP", 100, int)
    )
    """Token overlap applied when slicing text for HyperGraphRAG."""

    enable_hyper_entity_extraction: bool = field(
        default=get_env_value("ENABLE_HYPER_ENTITY_EXTRACTION", True, bool)
    )
    """If false, skip HyperGraphRAG chunk ingestion/extraction and rely on GraphCore graph."""

    bltcy_api_key: str = field(default=get_env_value("BLTCY_API_KEY", "", str))
    """API key used for OpenAI-compatible chat completion calls."""

    bltcy_api_base: str = field(
        default=get_env_value(
            "BLTCY_API_BASE",
            "https://api.openai.com/v1",
            str,
        )
    )
    """Endpoint base for an OpenAI-compatible API."""

    bltcy_model: str = field(
        default=get_env_value("BLTCY_MODEL", "qwen-plus", str)
    )
    """Model name used when calling the OpenAI-compatible API."""

    # Knowledge Graph Configuration
    # ---
    knowledge_graph_storage: str = field(default=get_env_value("KNOWLEDGE_GRAPH_STORAGE", "sqlite", str))
    """Knowledge graph storage backend: 'memory', 'file', or 'sqlite'."""

    knowledge_graph_path: str = field(default=get_env_value("KNOWLEDGE_GRAPH_PATH", "./knowledge_graph.db", str))
    """Path to knowledge graph storage file/database."""

    entity_disambiguation_threshold: float = field(default=get_env_value("ENTITY_DISAMBIGUATION_THRESHOLD", 0.6, float))
    """Threshold for entity disambiguation (0.0-1.0)."""

    relationship_validation_threshold: float = field(default=get_env_value("RELATIONSHIP_VALIDATION_THRESHOLD", 0.8, float))
    """Threshold for automatic relationship validation (0.0-1.0)."""

    enable_relationship_auto_validation: bool = field(default=get_env_value("ENABLE_RELATIONSHIP_AUTO_VALIDATION", True, bool))
    """Enable automatic relationship validation."""

    # UI Configuration
    # ---
    ui_enabled: bool = field(default=get_env_value("UI_ENABLED", True, bool))
    """Enable web UI for testing and management."""

    ui_port: int = field(default=get_env_value("UI_PORT", 5000, int))
    """Port for web UI server."""

    ui_host: str = field(default=get_env_value("UI_HOST", "0.0.0.0", str))
    """Host for web UI server."""

    enable_api_endpoints: bool = field(default=get_env_value("ENABLE_API_ENDPOINTS", True, bool))
    """Enable API endpoints for programmatic access."""

    # Visualization Configuration
    # ---
    visualization_max_entities: int = field(default=get_env_value("VISUALIZATION_MAX_ENTITIES", 100, int))
    """Maximum number of entities to include in visualization."""

    visualization_max_relationships: int = field(default=get_env_value("VISUALIZATION_MAX_RELATIONSHIPS", 200, int))
    """Maximum number of relationships to include in visualization."""

    # Version Control Configuration
    # ---
    enable_version_control: bool = field(default=get_env_value("ENABLE_VERSION_CONTROL", True, bool))
    """Enable knowledge graph version control."""

    snapshots_directory: str = field(default=get_env_value("SNAPSHOTS_DIRECTORY", "./snapshots", str))
    """Directory for storing knowledge graph snapshots."""

    auto_snapshot_interval: int = field(default=get_env_value("AUTO_SNAPSHOT_INTERVAL", 3600, int))
    """Interval for automatic snapshots in seconds (0 to disable)."""

    def __post_init__(self):
        """Post-initialization setup for backward compatibility"""
        # Support legacy environment variable names for backward compatibility
        legacy_parse_method = get_env_value("MINERU_PARSE_METHOD", None, str)
        if legacy_parse_method and not get_env_value("PARSE_METHOD", None, str):
            self.parse_method = legacy_parse_method
            import warnings

            warnings.warn(
                "MINERU_PARSE_METHOD is deprecated. Use PARSE_METHOD instead.",
                DeprecationWarning,
                stacklevel=2,
            )

    @property
    def mineru_parse_method(self) -> str:
        """
        Backward compatibility property for old code.

        .. deprecated::
           Use `parse_method` instead. This property will be removed in a future version.
        """
        import warnings

        warnings.warn(
            "mineru_parse_method is deprecated. Use parse_method instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.parse_method

    @mineru_parse_method.setter
    def mineru_parse_method(self, value: str):
        """Setter for backward compatibility"""
        import warnings

        warnings.warn(
            "mineru_parse_method is deprecated. Use parse_method instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.parse_method = value
