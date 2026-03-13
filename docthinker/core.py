"""
Complete document parsing + multimodal content insertion Pipeline

This script integrates:
1. Document parsing (using configurable parsers)
2. Pure text content GraphCore insertion
3. Specialized processing for multimodal content (using different processors)
"""
# 核心流程：解析-插入-查询-批量处理。
import os
from typing import Dict, Any, Optional, Callable, List
import sys
import asyncio
import atexit
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

from docthinker.knowledge_graph import KnowledgeGraph
from docthinker.knowledge_base import KnowledgeBaseManager
from docthinker.knowledge_base_storage import KnowledgeBaseStorage

# Add project root directory to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables from .env file BEFORE importing GraphCore
# This is critical for TIKTOKEN_CACHE_DIR to work properly in offline environments
# The OS environment variables take precedence over the .env file
load_dotenv(dotenv_path=".env", override=False)

from graphcore.coregraph import GraphCore
from graphcore.coregraph.utils import logger

# Import configuration and modules
from docthinker.config import DocThinkerConfig
from docthinker.query import QueryMixin
from docthinker.processor import ProcessorMixin
from docthinker.batch import BatchMixin
from docthinker.utils import get_processor_supports
from docthinker.parser import MineruParser, DoclingParser
from docthinker.twi_adapter import ThinkWithImageRunner

# Import specialized processors
from docthinker.modalprocessors import (
    ImageModalProcessor,
    TableModalProcessor,
    EquationModalProcessor,
    GenericModalProcessor,
    ContextExtractor,
    ContextConfig,
)


@dataclass
class DocThinker(QueryMixin, ProcessorMixin, BatchMixin):
    """Multimodal Document Processing Pipeline - Complete document parsing and insertion pipeline"""

    # Core Components
    # ---
    graphcore: Optional[GraphCore] = field(default=None)
    """Optional pre-initialized GraphCore instance."""

    llm_model_func: Optional[Callable] = field(default=None)
    """LLM model function for text analysis."""

    keyword_llm_model_func: Optional[Callable] = field(default=None)
    """Lightweight LLM for keyword extraction. Falls back to llm_model_func if None."""

    entity_extraction_llm_model_func: Optional[Callable] = field(default=None)
    """Lightweight LLM for entity extraction during ingestion. Falls back to llm_model_func if None."""

    vision_model_func: Optional[Callable] = field(default=None)
    """Vision model function for image analysis."""

    embedding_func: Optional[Callable] = field(default=None)
    """Embedding function for text vectorization."""

    config: Optional[DocThinkerConfig] = field(default=None)
    """Configuration object, if None will create with environment variables."""

    # GraphCore Configuration
    # ---
    graphcore_kwargs: Dict[str, Any] = field(default_factory=dict)
    """Additional keyword arguments for GraphCore initialization when graphcore is not provided.
    This allows passing all GraphCore configuration parameters like:
    - kv_storage, vector_storage, graph_storage, doc_status_storage
    - top_k, chunk_top_k, max_entity_tokens, max_relation_tokens, max_total_tokens
    - cosine_threshold, related_chunk_number
    - chunk_token_size, chunk_overlap_token_size, tokenizer, tiktoken_model_name
    - embedding_batch_num, embedding_func_max_async, embedding_cache_config
    - llm_model_name, llm_model_max_token_size, llm_model_max_async, llm_model_kwargs
    - rerank_model_func, vector_db_storage_cls_kwargs, enable_llm_cache
    - max_parallel_insert, max_graph_nodes, addon_params, etc.
    """

    # Internal State
    # ---
    modal_processors: Dict[str, Any] = field(default_factory=dict, init=False)
    """Dictionary of multimodal processors."""

    context_extractor: Optional[ContextExtractor] = field(default=None, init=False)
    """Context extractor for providing surrounding content to modal processors."""

    parse_cache: Optional[Any] = field(default=None, init=False)
    """Parse result cache storage using GraphCore KV storage."""

    _parser_installation_checked: bool = field(default=False, init=False)
    """Flag to track if parser installation has been checked."""

    hyper_chunk_sink: Optional[Callable[..., Any]] = field(default=None, init=False)
    """Optional callback used by the auto-thinking orchestrator to collect chunks."""

    knowledge_graph: Optional[KnowledgeGraph] = field(default=None, init=False)
    """Knowledge graph for managing entities and relationships across documents."""

    graph_path: Optional[Path] = field(default=None, init=False)
    """Path to save/load knowledge graph."""

    knowledge_base_manager: Optional[KnowledgeBaseManager] = field(default=None, init=False)
    """Manager for multiple knowledge bases."""

    knowledge_base_storage: Optional[KnowledgeBaseStorage] = field(default=None, init=False)
    """SQLite-based storage for knowledge bases."""

    use_knowledge_base: bool = field(default=True, init=False)
    """Flag to enable/disable knowledge base functionality."""

    def __post_init__(self):
        """Post-initialization setup following GraphCore pattern"""
        # Initialize configuration if not provided
        if self.config is None:
            self.config = DocThinkerConfig()

        # Set working directory
        self.working_dir = self.config.working_dir

        # Set up logger (use existing logger, don't configure it)
        self.logger = logger

        # Set up document parser
        self.doc_parser = (
            DoclingParser() if self.config.parser == "docling" else MineruParser()
        )

        # Register close method for cleanup
        atexit.register(self.close)

        # Create working directory if needed
        if not os.path.exists(self.working_dir):
            os.makedirs(self.working_dir)
            self.logger.info(f"Created working directory: {self.working_dir}")

        # Log configuration info
        self.logger.info("DocThinker initialized with config:")
        self.logger.info(f"  Working directory: {self.config.working_dir}")
        self.logger.info(f"  Parser: {self.config.parser}")
        self.logger.info(f"  Parse method: {self.config.parse_method}")
        self.logger.info(
            f"  Multimodal processing - Image: {self.config.enable_image_processing}, "
            f"Table: {self.config.enable_table_processing}, "
            f"Equation: {self.config.enable_equation_processing}"
        )
        self.logger.info(f"  Max concurrent files: {self.config.max_concurrent_files}")

        if getattr(self, "twi_runner", None) is None:
            self.twi_runner = ThinkWithImageRunner()
        
        # Initialize knowledge graph
        self._initialize_knowledge_graph()
        
        # Initialize knowledge base components
        self._initialize_knowledge_base()

    def close(self):
        """Cleanup resources when object is destroyed"""
        try:
            import asyncio

            if asyncio.get_event_loop().is_running():
                # If we're in an async context, schedule cleanup
                asyncio.create_task(self.finalize_storages())
            else:
                # Run cleanup synchronously
                asyncio.run(self.finalize_storages())
        except Exception as e:
            # Use print instead of logger since logger might be cleaned up already
            print(f"Warning: Failed to finalize DocThinker storages: {e}")

    def _create_context_config(self) -> ContextConfig:
        """Create context configuration from DocThinker config"""
        return ContextConfig(
            context_window=self.config.context_window,
            context_mode=self.config.context_mode,
            max_context_tokens=self.config.max_context_tokens,
            include_headers=self.config.include_headers,
            include_captions=self.config.include_captions,
            filter_content_types=self.config.context_filter_content_types,
        )

    def _create_context_extractor(self) -> ContextExtractor:
        """Create context extractor with tokenizer from GraphCore"""
        if self.graphcore is None:
            raise ValueError(
                "GraphCore must be initialized before creating context extractor"
            )

        context_config = self._create_context_config()
        return ContextExtractor(
            config=context_config, tokenizer=self.graphcore.tokenizer
        )

    def _initialize_processors(self):
        """Initialize multimodal processors with appropriate model functions"""
        if self.graphcore is None:
            raise ValueError(
                "GraphCore instance must be initialized before creating processors"
            )

        # Create context extractor
        self.context_extractor = self._create_context_extractor()

        # Create different multimodal processors based on configuration
        self.modal_processors = {}

        if self.config.enable_image_processing:
            self.modal_processors["image"] = ImageModalProcessor(
                graphcore=self.graphcore,
                modal_caption_func=self.vision_model_func or self.llm_model_func,
                context_extractor=self.context_extractor,
            )

        if self.config.enable_table_processing:
            self.modal_processors["table"] = TableModalProcessor(
                graphcore=self.graphcore,
                modal_caption_func=self.llm_model_func,
                context_extractor=self.context_extractor,
            )

        if self.config.enable_equation_processing:
            self.modal_processors["equation"] = EquationModalProcessor(
                graphcore=self.graphcore,
                modal_caption_func=self.llm_model_func,
                context_extractor=self.context_extractor,
            )

        # Always include generic processor as fallback
        self.modal_processors["generic"] = GenericModalProcessor(
            graphcore=self.graphcore,
            modal_caption_func=self.llm_model_func,
            context_extractor=self.context_extractor,
        )

        self.logger.info("Multimodal processors initialized with context support")
        self.logger.info(f"Available processors: {list(self.modal_processors.keys())}")
        self.logger.info(f"Context configuration: {self._create_context_config()}")

    def update_config(self, **kwargs):
        """Update configuration with new values"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                self.logger.debug(f"Updated config: {key} = {value}")
            else:
                self.logger.warning(f"Unknown config parameter: {key}")

    async def _ensure_graphcore_initialized(self):
        """Ensure GraphCore instance is initialized, create if necessary"""
        try:
            # Check parser installation first
            if not self._parser_installation_checked:
                if not self.doc_parser.check_installation():
                    error_msg = (
                        f"Parser '{self.config.parser}' is not properly installed. "
                        "Please install it using 'pip install' or 'uv pip install'."
                    )
                    self.logger.error(error_msg)
                    return {"success": False, "error": error_msg}

                self._parser_installation_checked = True
                self.logger.info(f"Parser '{self.config.parser}' installation verified")

            if self.graphcore is not None:
                # GraphCore was pre-provided, but we need to ensure it's properly initialized
                try:
                    # Ensure GraphCore storages are initialized
                    if (
                        not hasattr(self.graphcore, "_storages_status")
                        or self.graphcore._storages_status.name != "INITIALIZED"
                    ):
                        self.logger.info(
                            "Initializing storages for pre-provided GraphCore instance"
                        )
                        await self.graphcore.initialize_storages()
                        from graphcore.coregraph.kg.shared_storage import (
                            initialize_pipeline_status,
                        )

                        await initialize_pipeline_status()

                    # Initialize parse cache if not already done
                    if self.parse_cache is None:
                        self.logger.info(
                            "Initializing parse cache for pre-provided GraphCore instance"
                        )
                        self.parse_cache = (
                            self.graphcore.key_string_value_json_storage_cls(
                                namespace="parse_cache",
                                workspace=self.graphcore.workspace,
                                global_config=self.graphcore.__dict__,
                                embedding_func=self.embedding_func,
                            )
                        )
                        await self.parse_cache.initialize()

                    # Initialize processors if not already done
                    if not self.modal_processors:
                        self._initialize_processors()

                    return {"success": True}

                except Exception as e:
                    error_msg = (
                        f"Failed to initialize pre-provided GraphCore instance: {str(e)}"
                    )
                    self.logger.error(error_msg, exc_info=True)
                    return {"success": False, "error": error_msg}

            # Validate required functions for creating new GraphCore instance
            if self.llm_model_func is None:
                error_msg = "llm_model_func must be provided when GraphCore is not pre-initialized"
                self.logger.error(error_msg)
                return {"success": False, "error": error_msg}

            if self.embedding_func is None:
                error_msg = "embedding_func must be provided when GraphCore is not pre-initialized"
                self.logger.error(error_msg)
                return {"success": False, "error": error_msg}

            from graphcore.coregraph.kg.shared_storage import initialize_pipeline_status

            graphcore_params = {
                "working_dir": self.working_dir,
                "llm_model_func": self.llm_model_func,
                "embedding_func": self.embedding_func,
            }
            if self.keyword_llm_model_func is not None:
                graphcore_params["keyword_llm_model_func"] = self.keyword_llm_model_func
            if self.entity_extraction_llm_model_func is not None:
                graphcore_params["entity_extraction_llm_model_func"] = self.entity_extraction_llm_model_func

            # Merge user-provided graphcore_kwargs, which can override defaults
            graphcore_params.update(self.graphcore_kwargs)

            # Log the parameters being used for initialization (excluding sensitive data)
            log_params = {
                k: v
                for k, v in graphcore_params.items()
                if not callable(v)
                and k not in ["llm_model_kwargs", "vector_db_storage_cls_kwargs"]
            }
            self.logger.info(f"Initializing GraphCore with parameters: {log_params}")

            try:
                # Create GraphCore instance with merged parameters
                self.graphcore = GraphCore(**graphcore_params)
                try:
                    mpi = graphcore_params.get("max_parallel_insert")
                    if mpi is not None:
                        setattr(self.graphcore, "max_parallel_insert", int(mpi))
                except Exception:
                    pass
                await self.graphcore.initialize_storages()
                await initialize_pipeline_status()

                # Initialize parse cache storage using GraphCore's KV storage
                self.parse_cache = self.graphcore.key_string_value_json_storage_cls(
                    namespace="parse_cache",
                    workspace=self.graphcore.workspace,
                    global_config=self.graphcore.__dict__,
                    embedding_func=self.embedding_func,
                )
                await self.parse_cache.initialize()

                # Initialize processors after GraphCore is ready
                self._initialize_processors()

                self.logger.info(
                    "GraphCore, parse cache, and multimodal processors initialized"
                )
                return {"success": True}

            except Exception as e:
                error_msg = f"Failed to initialize GraphCore instance: {str(e)}"
                self.logger.error(error_msg, exc_info=True)
                return {"success": False, "error": error_msg}

        except Exception as e:
            error_msg = f"Unexpected error during GraphCore initialization: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return {"success": False, "error": error_msg}

    async def finalize_storages(self):
        """Finalize all storages including parse cache and GraphCore storages

        This method should be called when shutting down to properly clean up resources
        and persist any cached data. It will finalize both the parse cache and GraphCore's
        internal storages.

        Example usage:
            try:
                rag_anything = DocThinker(...)
                await rag_anything.process_file("document.pdf")
                # ... other operations ...
            finally:
                # Always finalize storages to clean up resources
                if rag_anything:
                    await rag_anything.finalize_storages()

        Note:
            - This method is automatically called in __del__ when the object is destroyed
            - Manual calling is recommended in production environments
            - All finalization tasks run concurrently for better performance
        """
        try:
            tasks = []

            # Finalize parse cache if it exists
            if self.parse_cache is not None:
                tasks.append(self.parse_cache.finalize())
                self.logger.debug("Scheduled parse cache finalization")

            # Finalize GraphCore storages if GraphCore is initialized
            if self.graphcore is not None:
                tasks.append(self.graphcore.finalize_storages())
                self.logger.debug("Scheduled GraphCore storages finalization")

            # Finalize knowledge graph if it exists
            if self.knowledge_graph is not None and self.graph_path is not None:
                self.knowledge_graph.save(str(self.graph_path))
                self.logger.debug("Saved knowledge graph")

            # Run all finalization tasks concurrently
            if tasks:
                await asyncio.gather(*tasks)
                self.logger.info("Successfully finalized all DocThinker storages")
            else:
                self.logger.debug("No storages to finalize")

        except Exception as e:
            self.logger.error(f"Error during storage finalization: {e}")
            raise

    def check_parser_installation(self) -> bool:
        """
        Check if the configured parser is properly installed

        Returns:
            bool: True if the configured parser is properly installed
        """
        return self.doc_parser.check_installation()

    def verify_parser_installation_once(self) -> bool:
        if not self._parser_installation_checked:
            if not self.doc_parser.check_installation():
                raise RuntimeError(
                    f"Parser '{self.config.parser}' is not properly installed. "
                    "Please install it using pip install or uv pip install."
                )
            self._parser_installation_checked = True
            self.logger.info(f"Parser '{self.config.parser}' installation verified")
        return True

    def _initialize_knowledge_graph(self):
        """Initialize knowledge graph"""
        self.graph_path = Path(self.working_dir) / "knowledge_graph.json"
        try:
            self.knowledge_graph = KnowledgeGraph.load(str(self.graph_path))
        except Exception:
            self.knowledge_graph = KnowledgeGraph()
        self.logger.info(f"Knowledge graph initialized, loaded {len(self.knowledge_graph.entities)} entities and {len(self.knowledge_graph.relationships)} relationships")
        
        # Create file-based storage backend
        from docthinker.knowledge_graph import FileKnowledgeGraphStorage
        file_storage = FileKnowledgeGraphStorage(str(self.graph_path))
        
        # Initialize storage for knowledge graph
        self.knowledge_graph.initialize_storage(file_storage, load_from_storage=False)
        self.logger.info(f"Knowledge graph storage initialized with file backend: {self.graph_path}")
    
    def _initialize_knowledge_base(self):
        """Initialize knowledge base components"""
        if not self.use_knowledge_base:
            self.logger.info("Knowledge base functionality is disabled")
            return
        
        try:
            # Initialize knowledge base storage
            kb_db_path = Path(self.working_dir) / "knowledge_base.db"
            self.knowledge_base_storage = KnowledgeBaseStorage(str(kb_db_path))
            self.logger.info(f"Knowledge base storage initialized at {kb_db_path}")
            
            # Initialize knowledge base manager
            kb_storage_path = Path(self.working_dir) / "knowledge_bases"
            self.knowledge_base_manager = KnowledgeBaseManager(str(kb_storage_path))
            self.logger.info(f"Knowledge base manager initialized with storage path {kb_storage_path}")
            
            # Create global knowledge base if it doesn't exist
            if not self.knowledge_base_manager.get_knowledge_base("global"):
                self.knowledge_base_manager.create_knowledge_base("global", "global")
                self.logger.info("Created global knowledge base")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize knowledge base components: {e}")
            self.use_knowledge_base = False

    def get_config_info(self) -> Dict[str, Any]:
        """Get current configuration information"""
        config_info = {
            "directory": {
                "working_dir": self.config.working_dir,
                "parser_output_dir": self.config.parser_output_dir,
            },
            "parsing": {
                "parser": self.config.parser,
                "parse_method": self.config.parse_method,
                "display_content_stats": self.config.display_content_stats,
            },
            "multimodal_processing": {
                "enable_image_processing": self.config.enable_image_processing,
                "enable_table_processing": self.config.enable_table_processing,
                "enable_equation_processing": self.config.enable_equation_processing,
            },
            "context_extraction": {
                "context_window": self.config.context_window,
                "context_mode": self.config.context_mode,
                "max_context_tokens": self.config.max_context_tokens,
                "include_headers": self.config.include_headers,
                "include_captions": self.config.include_captions,
                "filter_content_types": self.config.context_filter_content_types,
            },
            "batch_processing": {
                "max_concurrent_files": self.config.max_concurrent_files,
                "supported_file_extensions": self.config.supported_file_extensions,
                "recursive_folder_processing": self.config.recursive_folder_processing,
            },
            "logging": {
                "note": "Logging fields have been removed - configure logging externally",
            },
        }

        # Add GraphCore configuration if available
        if self.graphcore_kwargs:
            # Filter out sensitive data and callable objects for display
            safe_kwargs = {
                k: v
                for k, v in self.graphcore_kwargs.items()
                if not callable(v)
                and k not in ["llm_model_kwargs", "vector_db_storage_cls_kwargs"]
            }
            config_info["graphcore_config"] = {
                "custom_parameters": safe_kwargs,
                "note": "GraphCore will be initialized with these additional parameters",
            }
        else:
            config_info["graphcore_config"] = {
                "custom_parameters": {},
                "note": "Using default GraphCore parameters",
            }

        return config_info

    def set_content_source_for_context(
        self, content_source, content_format: str = "auto"
    ):
        """Set content source for context extraction in all modal processors

        Args:
            content_source: Source content for context extraction (e.g., MinerU content list)
            content_format: Format of content source ("minerU", "text_chunks", "auto")
        """
        if not self.modal_processors:
            self.logger.warning(
                "Modal processors not initialized. Content source will be set when processors are created."
            )
            return

        for processor_name, processor in self.modal_processors.items():
            try:
                processor.set_content_source(content_source, content_format)
                self.logger.debug(f"Set content source for {processor_name} processor")
            except Exception as e:
                self.logger.error(
                    f"Failed to set content source for {processor_name}: {e}"
                )

        self.logger.info(
            f"Content source set for context extraction (format: {content_format})"
        )

    def update_context_config(self, **context_kwargs):
        """Update context extraction configuration

        Args:
            **context_kwargs: Context configuration parameters to update
                (context_window, context_mode, max_context_tokens, etc.)
        """
        # Update the main config
        for key, value in context_kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                self.logger.debug(f"Updated context config: {key} = {value}")
            else:
                self.logger.warning(f"Unknown context config parameter: {key}")

        # Recreate context extractor with new config if processors are initialized
        if self.graphcore and self.modal_processors:
            try:
                self.context_extractor = self._create_context_extractor()
                # Update all processors with new context extractor
                for processor_name, processor in self.modal_processors.items():
                    processor.context_extractor = self.context_extractor

                self.logger.info(
                    "Context configuration updated and applied to all processors"
                )
                self.logger.info(
                    f"New context configuration: {self._create_context_config()}"
                )
            except Exception as e:
                self.logger.error(f"Failed to update context configuration: {e}")

    def get_processor_info(self) -> Dict[str, Any]:
        """Get processor information"""
        base_info = {
            "mineru_installed": MineruParser.check_installation(MineruParser()),
            "config": self.get_config_info(),
            "models": {
                "llm_model": "External function"
                if self.llm_model_func
                else "Not provided",
                "vision_model": "External function"
                if self.vision_model_func
                else "Not provided",
                "embedding_model": "External function"
                if self.embedding_func
                else "Not provided",
            },
            "knowledge_base": {
                "enabled": self.use_knowledge_base,
                "manager_initialized": self.knowledge_base_manager is not None,
                "storage_initialized": self.knowledge_base_storage is not None,
            }
        }

        if not self.modal_processors:
            base_info["status"] = "Not initialized"
            base_info["processors"] = {}
        else:
            base_info["status"] = "Initialized"
            base_info["processors"] = {}

            for proc_type, processor in self.modal_processors.items():
                base_info["processors"][proc_type] = {
                    "class": processor.__class__.__name__,
                    "supports": get_processor_supports(proc_type),
                    "enabled": True,
                }

        return base_info
    
    def add_knowledge_entry(self, content: str, entry_type: str = "generic", 
                           metadata: Optional[Dict[str, Any]] = None, kb_name: str = "global") -> Optional[str]:
        """Add a knowledge entry to a knowledge base"""
        if not self.use_knowledge_base or not self.knowledge_base_manager:
            return None
        
        try:
            return self.knowledge_base_manager.add_entry_content_to_kb(
                kb_name=kb_name,
                content=content,
                entry_type=entry_type,
                metadata=metadata
            )
        except Exception as e:
            self.logger.error(f"Failed to add knowledge entry: {e}")
            return None
    
    def add_document_knowledge_entry(self, doc_id: str, content: str, 
                                  entry_type: str = "document", 
                                  metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Add a knowledge entry to a document-specific knowledge base"""
        if not self.use_knowledge_base or not self.knowledge_base_manager:
            return None
        
        try:
            # Create document knowledge base if it doesn't exist
            doc_kb = self.knowledge_base_manager.get_document_kb(doc_id)
            if not doc_kb:
                doc_kb = self.knowledge_base_manager.create_document_kb(doc_id)
            
            return self.knowledge_base_manager.add_entry_content_to_kb(
                kb_name=doc_kb.name,
                content=content,
                entry_type=entry_type,
                metadata=metadata
            )
        except Exception as e:
            self.logger.error(f"Failed to add document knowledge entry: {e}")
            return None

    def add_cognitive_memory(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        *,
        source_type: str = "text",
        session_id: Optional[str] = None,
    ) -> Optional[str]:
        if not self.use_knowledge_base or not self.knowledge_base_manager:
            return None
        payload = metadata or {}
        summary = payload.get("summary") or ""
        content = summary.strip() if isinstance(summary, str) else ""
        if not content:
            content = (text or "").strip()
        if len(content) > 2000:
            content = f"{content[:2000]}..."
        entry_metadata = {
            "source_type": source_type,
            "session_id": session_id,
            "summary": payload.get("summary"),
            "reasoning": payload.get("reasoning"),
            "key_points": payload.get("key_points") or [],
            "concepts": payload.get("concepts") or [],
            "hypotheses": payload.get("hypotheses") or [],
            "action_items": payload.get("action_items") or [],
        }
        entry_id = self.add_knowledge_entry(
            content=content,
            entry_type="memory",
            metadata=entry_metadata,
            kb_name="global",
        )
        if not entry_id:
            return None
        try:
            from docthinker.hypergraph.utils import compute_mdhash_id

            doc_id = compute_mdhash_id(text or content, prefix="doc-")
            self.add_document_knowledge_entry(
                doc_id=doc_id,
                content=content,
                entry_type="memory",
                metadata=entry_metadata,
            )
        except Exception:
            pass
        if self.knowledge_graph and payload.get("entities"):
            for entity in payload.get("entities") or []:
                if not isinstance(entity, dict):
                    continue
                name = (entity.get("name") or "").strip()
                if not name:
                    continue
                entity_type = entity.get("entity_type")
                target = self.knowledge_graph.get_entity_by_name(
                    name, entity_type
                ) or self.knowledge_graph.get_entity_by_name(name)
                if not target:
                    continue
                try:
                    self.link_entity_to_knowledge_entry("global", target.id, entry_id)
                except Exception:
                    continue
        return entry_id
    
    def query_knowledge_base(self, query_text: str, kb_name: str = "global", 
                           entry_types: Optional[List[str]] = None) -> List[Any]:
        """Query a knowledge base"""
        if not self.use_knowledge_base or not self.knowledge_base_manager:
            return []
        
        try:
            return self.knowledge_base_manager.query_knowledge_base(
                kb_name=kb_name,
                query_text=query_text,
                entry_types=entry_types
            )
        except Exception as e:
            self.logger.error(f"Failed to query knowledge base: {e}")
            return []
    
    def query_all_knowledge_bases(self, query_text: str, 
                               kb_types: Optional[List[str]] = None, 
                               entry_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Query all knowledge bases"""
        if not self.use_knowledge_base or not self.knowledge_base_manager:
            return []
        
        try:
            return self.knowledge_base_manager.query_all_knowledge_bases(
                query_text=query_text,
                kb_types=kb_types,
                entry_types=entry_types
            )
        except Exception as e:
            self.logger.error(f"Failed to query all knowledge bases: {e}")
            return []
    
    def create_knowledge_base(self, name: str, kb_type: str, 
                            metadata: Optional[Dict[str, Any]] = None) -> Optional[Any]:
        """Create a new knowledge base"""
        if not self.use_knowledge_base or not self.knowledge_base_manager:
            return None
        
        try:
            return self.knowledge_base_manager.create_knowledge_base(
                name=name,
                kb_type=kb_type,
                metadata=metadata
            )
        except Exception as e:
            self.logger.error(f"Failed to create knowledge base: {e}")
            return None
    
    def get_knowledge_base(self, name: str) -> Optional[Any]:
        """Get a knowledge base by name"""
        if not self.use_knowledge_base or not self.knowledge_base_manager:
            return None
        
        try:
            return self.knowledge_base_manager.get_knowledge_base(name)
        except Exception as e:
            self.logger.error(f"Failed to get knowledge base: {e}")
            return None
    
    def query_knowledge_base_with_reasoning(self, kb_name: str, query_text: str, 
                                           entry_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Query a knowledge base with reasoning capabilities"""
        if not self.use_knowledge_base or not self.knowledge_base_manager:
            return []
        
        try:
            results, reasoning_info = self.knowledge_base_manager.query_knowledge_base_with_reasoning(
                kb_name=kb_name,
                query_text=query_text,
                entry_types=entry_types
            )
            
            # Format results with kb info
            formatted_results = []
            for entry in results:
                formatted_results.append({
                    "kb_name": kb_name,
                    "entry": entry
                })
            
            return formatted_results, reasoning_info
        except Exception as e:
            self.logger.error(f"Failed to query knowledge base with reasoning: {e}")
            return [], {}
    
    def multi_dimension_query(self, kb_name: str, 
                            query_text: Optional[str] = None, 
                            entry_types: Optional[List[str]] = None,
                            entities: Optional[List[str]] = None,
                            metadata_filters: Optional[Dict[str, Any]] = None,
                            min_confidence: float = 0.0) -> List[Dict[str, Any]]:
        """Perform multi-dimension query on a knowledge base"""
        if not self.use_knowledge_base or not self.knowledge_base_manager:
            return []
        
        try:
            results = self.knowledge_base_manager.multi_dimension_query_kb(
                kb_name=kb_name,
                query_text=query_text,
                entry_types=entry_types,
                entities=entities,
                metadata_filters=metadata_filters,
                min_confidence=min_confidence
            )
            
            # Format results with kb info
            formatted_results = []
            for entry in results:
                formatted_results.append({
                    "kb_name": kb_name,
                    "entry": entry
                })
            
            return formatted_results
        except Exception as e:
            self.logger.error(f"Failed to perform multi-dimension query: {e}")
            return []
    
    def multi_dimension_query_all(self, 
                                query_text: Optional[str] = None, 
                                kb_types: Optional[List[str]] = None,
                                entry_types: Optional[List[str]] = None,
                                entities: Optional[List[str]] = None,
                                metadata_filters: Optional[Dict[str, Any]] = None,
                                min_confidence: float = 0.0) -> List[Dict[str, Any]]:
        """Perform multi-dimension query on all knowledge bases"""
        if not self.use_knowledge_base or not self.knowledge_base_manager:
            return []
        
        try:
            return self.knowledge_base_manager.multi_dimension_query_all_kbs(
                query_text=query_text,
                kb_types=kb_types,
                entry_types=entry_types,
                entities=entities,
                metadata_filters=metadata_filters,
                min_confidence=min_confidence
            )
        except Exception as e:
            self.logger.error(f"Failed to perform multi-dimension query on all KBs: {e}")
            return []
    
    def sync_knowledge_bases_with_graph(self):
        """Sync all knowledge bases with the knowledge graph"""
        if not self.use_knowledge_base or not self.knowledge_base_manager or not self.knowledge_graph:
            return
        
        try:
            self.knowledge_base_manager.sync_all_kbs_with_knowledge_graph(self.knowledge_graph)
            self.logger.info("Synced all knowledge bases with knowledge graph")
        except Exception as e:
            self.logger.error(f"Failed to sync knowledge bases with graph: {e}")
    
    def link_entity_to_knowledge_entry(self, kb_name: str, entity_id: str, entry_id: str):
        """Link a knowledge graph entity to a knowledge entry"""
        if not self.use_knowledge_base or not self.knowledge_base_manager:
            return
        
        try:
            self.knowledge_base_manager.link_entity_to_entry(kb_name, entity_id, entry_id)
            self.logger.debug(f"Linked entity {entity_id} to entry {entry_id} in KB {kb_name}")
        except Exception as e:
            self.logger.error(f"Failed to link entity to entry: {e}")
