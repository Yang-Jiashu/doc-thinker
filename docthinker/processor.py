"""
Document processing functionality for DocThinker

Contains methods for parsing documents and processing multimodal content
"""
#主要负责chunk的切分，多模态信息的处理，关系实体抽�?import os
import time
import hashlib
import json
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path

from docthinker.base import DocStatus
from docthinker.parser import MineruParser, DoclingParser, MineruExecutionError
from docthinker.utils import (
    separate_content,
    insert_text_content,
    insert_text_content_with_multimodal_content,
    get_processor_for_type,
)
import asyncio
from graphcore.coregraph.utils import compute_mdhash_id


class ProcessorMixin:
    """ProcessorMixin class containing document processing functionality for DocThinker"""

    def _generate_cache_key(
        self, file_path: Path, parse_method: str = None, **kwargs
    ) -> str:
        """
        Generate cache key based on file path and parsing configuration

        Args:
            file_path: Path to the file
            parse_method: Parse method used
            **kwargs: Additional parser parameters

        Returns:
            str: Cache key for the file and configuration
        """

        # Get file modification time
        mtime = file_path.stat().st_mtime

        # Create configuration dict for cache key
        config_dict = {
            "file_path": str(file_path.absolute()),
            "mtime": mtime,
            "parser": self.config.parser,
            "parse_method": parse_method or self.config.parse_method,
        }

        # Add relevant kwargs to config
        relevant_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k
            in [
                "lang",
                "device",
                "start_page",
                "end_page",
                "formula",
                "table",
                "backend",
                "source",
            ]
        }
        config_dict.update(relevant_kwargs)

        # Generate hash from config
        config_str = json.dumps(config_dict, sort_keys=True)
        cache_key = hashlib.md5(config_str.encode()).hexdigest()

        return cache_key

    def _generate_content_based_doc_id(self, content_list: List[Dict[str, Any]]) -> str:
        """
        Generate doc_id based on document content

        Args:
            content_list: Parsed content list

        Returns:
            str: Content-based document ID with doc- prefix
        """
        from graphcore.coregraph.utils import compute_mdhash_id

        # Extract key content for ID generation
        content_hash_data = []

        for item in content_list:
            if isinstance(item, dict):
                # For text content, use the text
                if item.get("type") == "text" and item.get("text"):
                    content_hash_data.append(item["text"].strip())
                # For other content types, use key identifiers
                elif item.get("type") == "image" and item.get("img_path"):
                    content_hash_data.append(f"image:{item['img_path']}")
                elif item.get("type") == "table" and item.get("table_body"):
                    content_hash_data.append(f"table:{item['table_body']}")
                elif item.get("type") == "equation" and item.get("text"):
                    content_hash_data.append(f"equation:{item['text']}")
                else:
                    # For other types, use string representation
                    content_hash_data.append(str(item))

        # Create a content signature
        content_signature = "\n".join(content_hash_data)

        # Generate doc_id from content
        doc_id = compute_mdhash_id(content_signature, prefix="doc-")

        return doc_id

    async def _get_cached_result(
        self, cache_key: str, file_path: Path, parse_method: str = None, **kwargs
    ) -> tuple[List[Dict[str, Any]], str] | None:
        """
        Get cached parsing result if available and valid

        Args:
            cache_key: Cache key to look up
            file_path: Path to the file for mtime check
            parse_method: Parse method used
            **kwargs: Additional parser parameters

        Returns:
            tuple[List[Dict[str, Any]], str] | None: (content_list, doc_id) or None if not found/invalid
        """
        if not hasattr(self, "parse_cache") or self.parse_cache is None:
            return None

        try:
            cached_data = await self.parse_cache.get_by_id(cache_key)
            if not cached_data:
                return None

            # Check file modification time
            current_mtime = file_path.stat().st_mtime
            cached_mtime = cached_data.get("mtime", 0)

            if current_mtime != cached_mtime:
                self.logger.debug(f"Cache invalid - file modified: {cache_key}")
                return None

            # Check parsing configuration
            cached_config = cached_data.get("parse_config", {})
            current_config = {
                "parser": self.config.parser,
                "parse_method": parse_method or self.config.parse_method,
            }

            # Add relevant kwargs to current config
            relevant_kwargs = {
                k: v
                for k, v in kwargs.items()
                if k
                in [
                    "lang",
                    "device",
                    "start_page",
                    "end_page",
                    "formula",
                    "table",
                    "backend",
                    "source",
                ]
            }
            current_config.update(relevant_kwargs)

            if cached_config != current_config:
                self.logger.debug(f"Cache invalid - config changed: {cache_key}")
                return None

            content_list = cached_data.get("content_list", [])
            doc_id = cached_data.get("doc_id")

            if content_list and doc_id:
                self.logger.debug(
                    f"Found valid cached parsing result for key: {cache_key}"
                )
                return content_list, doc_id
            else:
                self.logger.debug(
                    f"Cache incomplete - missing content or doc_id: {cache_key}"
                )
                return None

        except Exception as e:
            self.logger.warning(f"Error accessing parse cache: {e}")

        return None

    async def _store_cached_result(
        self,
        cache_key: str,
        content_list: List[Dict[str, Any]],
        doc_id: str,
        file_path: Path,
        parse_method: str = None,
        **kwargs,
    ) -> None:
        """
        Store parsing result in cache

        Args:
            cache_key: Cache key to store under
            content_list: Content list to cache
            doc_id: Content-based document ID
            file_path: Path to the file for mtime storage
            parse_method: Parse method used
            **kwargs: Additional parser parameters
        """
        if not hasattr(self, "parse_cache") or self.parse_cache is None:
            return

        try:
            # Get file modification time
            file_mtime = file_path.stat().st_mtime

            # Create parsing configuration
            parse_config = {
                "parser": self.config.parser,
                "parse_method": parse_method or self.config.parse_method,
            }

            # Add relevant kwargs to config
            relevant_kwargs = {
                k: v
                for k, v in kwargs.items()
                if k
                in [
                    "lang",
                    "device",
                    "start_page",
                    "end_page",
                    "formula",
                    "table",
                    "backend",
                    "source",
                ]
            }
            parse_config.update(relevant_kwargs)

            cache_data = {
                cache_key: {
                    "content_list": content_list,
                    "doc_id": doc_id,
                    "mtime": file_mtime,
                    "parse_config": parse_config,
                    "cached_at": time.time(),
                    "cache_version": "1.0",
                }
            }
            await self.parse_cache.upsert(cache_data)
            # Ensure data is persisted to disk
            await self.parse_cache.index_done_callback()
            self.logger.info(f"Stored parsing result in cache: {cache_key}")
        except Exception as e:
            self.logger.warning(f"Error storing to parse cache: {e}")

    async def parse_document(
        self,
        file_path: str,
        output_dir: str = None,
        parse_method: str = None,
        display_stats: bool = None,
        **kwargs,
    ) -> tuple[List[Dict[str, Any]], str]:
        """
        Parse document with caching support

        Args:
            file_path: Path to the file to parse
            output_dir: Output directory (defaults to config.parser_output_dir)
            parse_method: Parse method (defaults to config.parse_method)
            display_stats: Whether to display content statistics (defaults to config.display_content_stats)
            **kwargs: Additional parameters for parser (e.g., lang, device, start_page, end_page, formula, table, backend, source)

        Returns:
            tuple[List[Dict[str, Any]], str]: (content_list, doc_id)
        """
        # Use config defaults if not provided
        if output_dir is None:
            output_dir = self.config.parser_output_dir
        if parse_method is None:
            parse_method = self.config.parse_method
        if display_stats is None:
            display_stats = self.config.display_content_stats

        self.logger.info(f"Starting document parsing: {file_path}")

        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Generate cache key based on file and configuration
        cache_key = self._generate_cache_key(file_path, parse_method, **kwargs)

        # Check cache first
        cached_result = await self._get_cached_result(
            cache_key, file_path, parse_method, **kwargs
        )
        if cached_result is not None:
            content_list, doc_id = cached_result
            self.logger.info(f"Using cached parsing result for: {file_path}")
            if display_stats:
                self.logger.info(
                    f"* Total blocks in cached content_list: {len(content_list)}"
                )
            return content_list, doc_id

        # Choose appropriate parsing method based on file extension
        ext = file_path.suffix.lower()

        try:
            doc_parser = (
                DoclingParser() if self.config.parser == "docling" else MineruParser()
            )

            # Log parser and method information
            self.logger.info(
                f"Using {self.config.parser} parser with method: {parse_method}"
            )

            if ext in [".pdf"]:
                self.logger.info("Detected PDF file, using parser for PDF...")
                content_list = await asyncio.to_thread(
                    doc_parser.parse_pdf,
                    pdf_path=file_path,
                    output_dir=output_dir,
                    method=parse_method,
                    **kwargs,
                )
            elif ext in [".txt", ".md"]:
                self.logger.info("Detected text file, using parser for text...")
                if hasattr(doc_parser, "parse_text_file"):
                    content_list = await asyncio.to_thread(
                        doc_parser.parse_text_file,
                        text_path=file_path,
                        output_dir=output_dir,
                        **kwargs,
                    )
                else:
                    self.logger.info(
                        f"Using generic parser for {ext} file (method={parse_method})..."
                    )
                    content_list = await asyncio.to_thread(
                        doc_parser.parse_document,
                        file_path=file_path,
                        method=parse_method,
                        output_dir=output_dir,
                        **kwargs,
                    )
            elif ext in [
                ".jpg",
                ".jpeg",
                ".png",
                ".bmp",
                ".tiff",
                ".tif",
                ".gif",
                ".webp",
            ]:
                self.logger.info("Detected image file, using parser for images...")
                # Use the selected parser's image parsing capability
                if hasattr(doc_parser, "parse_image"):
                    content_list = await asyncio.to_thread(
                        doc_parser.parse_image,
                        image_path=file_path,
                        output_dir=output_dir,
                        **kwargs,
                    )
                else:
                    # Fallback to MinerU for image parsing if current parser doesn't support it
                    self.logger.warning(
                        f"{self.config.parser} parser doesn't support image parsing, falling back to MinerU"
                    )
                    content_list = MineruParser().parse_image(
                        image_path=file_path, output_dir=output_dir, **kwargs
                    )
            elif ext in [
                ".doc",
                ".docx",
                ".ppt",
                ".pptx",
                ".xls",
                ".xlsx",
                ".html",
                ".htm",
                ".xhtml",
            ]:
                self.logger.info(
                    "Detected Office or HTML document, using parser for Office/HTML..."
                )
                content_list = await asyncio.to_thread(
                    doc_parser.parse_office_doc,
                    doc_path=file_path,
                    output_dir=output_dir,
                    **kwargs,
                )
            else:
                # For other or unknown formats, use generic parser
                self.logger.info(
                    f"Using generic parser for {ext} file (method={parse_method})..."
                )
                content_list = await asyncio.to_thread(
                    doc_parser.parse_document,
                    file_path=file_path,
                    method=parse_method,
                    output_dir=output_dir,
                    **kwargs,
                )

        except MineruExecutionError as e:
            self.logger.error(f"Mineru command failed: {e}")
            raise
        except Exception as e:
            self.logger.error(
                f"Error during parsing with {self.config.parser} parser: {str(e)}"
            )
            raise e

        msg = f"Parsing {file_path} complete! Extracted {len(content_list)} content blocks"
        self.logger.info(msg)

        if len(content_list) == 0:
            raise ValueError("Parsing failed: No content was extracted")

        # Generate doc_id based on content
        doc_id = self._generate_content_based_doc_id(content_list)

        # Store result in cache
        await self._store_cached_result(
            cache_key, content_list, doc_id, file_path, parse_method, **kwargs
        )

        # Display content statistics if requested
        if display_stats:
            self.logger.info("\nContent Information:")
            self.logger.info(f"* Total blocks in content_list: {len(content_list)}")

            # Count elements by type
            block_types: Dict[str, int] = {}
            for block in content_list:
                if isinstance(block, dict):
                    block_type = block.get("type", "unknown")
                    if isinstance(block_type, str):
                        block_types[block_type] = block_types.get(block_type, 0) + 1

            self.logger.info("* Content block types:")
            for block_type, count in block_types.items():
                self.logger.info(f"  - {block_type}: {count}")

        return content_list, doc_id

    async def _process_multimodal_content(
        self,
        multimodal_items: List[Dict[str, Any]],
        file_path: str,
        doc_id: str,
        pipeline_status: Optional[Any] = None,
        pipeline_status_lock: Optional[Any] = None,
    ):
        """
        Process multimodal content (using specialized processors)

        Args:
            multimodal_items: List of multimodal items
            file_path: File path (for reference)
            doc_id: Document ID for proper chunk association
            pipeline_status: Pipeline status object
            pipeline_status_lock: Pipeline status lock
        """

        if not multimodal_items:
            self.logger.debug("No multimodal content to process")
            return

        # Check multimodal processing status - handle GraphCore's early DocStatus.PROCESSED marking
        try:
            existing_doc_status = await self.graphcore.doc_status.get_by_id(doc_id)
            if existing_doc_status:
                # Check if multimodal content is already processed
                multimodal_processed = existing_doc_status.get(
                    "multimodal_processed", False
                )

                if multimodal_processed:
                    self.logger.info(
                        f"Document {doc_id} multimodal content is already processed"
                    )
                    return

                # Even if status is DocStatus.PROCESSED (text processing done),
                # we still need to process multimodal content if not yet done
                doc_status = existing_doc_status.get("status", "")
                if doc_status == DocStatus.PROCESSED and not multimodal_processed:
                    self.logger.info(
                        f"Document {doc_id} text processing is complete, but multimodal content still needs processing"
                    )
                    # Continue with multimodal processing
                elif doc_status == DocStatus.PROCESSED and multimodal_processed:
                    self.logger.info(
                        f"Document {doc_id} is fully processed (text + multimodal)"
                    )
                    return

        except Exception as e:
            self.logger.debug(f"Error checking document status for {doc_id}: {e}")
            # Continue with processing if cache check fails

        # Use ProcessorMixin's own batch processing that can handle multiple content types
        log_message = "Starting multimodal content processing..."
        self.logger.info(log_message)
        if pipeline_status_lock and pipeline_status:
            async with pipeline_status_lock:
                pipeline_status["latest_message"] = log_message
                pipeline_status["history_messages"].append(log_message)

        attempts = 3
        for attempt in range(attempts):
            try:
                await self._ensure_graphcore.coregraph_initialized()
                await self._process_multimodal_content_batch_type_aware(
                    multimodal_items=multimodal_items, file_path=file_path, doc_id=doc_id
                )
                await self._mark_multimodal_processing_complete(doc_id)
                log_message = "Multimodal content processing complete"
                self.logger.info(log_message)
                if pipeline_status_lock and pipeline_status:
                    async with pipeline_status_lock:
                        pipeline_status["latest_message"] = log_message
                        pipeline_status["history_messages"].append(log_message)
                break
            except Exception as e:
                self.logger.error(f"Error in multimodal processing: {e}")
                if attempt < attempts - 1:
                    await asyncio.sleep(1 * (2 ** attempt))
                    continue
                self.logger.warning("Falling back to individual multimodal processing")
                await self._process_multimodal_content_individual(
                    multimodal_items, file_path, doc_id
                )
                await self._mark_multimodal_processing_complete(doc_id)

    async def _process_multimodal_content_individual(
        self, multimodal_items: List[Dict[str, Any]], file_path: str, doc_id: str
    ):
        """
        Process multimodal content individually (fallback method)

        Args:
            multimodal_items: List of multimodal items
            file_path: File path (for reference)
            doc_id: Document ID for proper chunk association
        """
        file_name = os.path.basename(file_path)

        # Collect all chunk results for batch processing (similar to text content processing)
        all_chunk_results = []
        multimodal_chunk_ids = []

        # Get current text chunks count to set proper order indexes for multimodal chunks
        existing_doc_status = await self.graphcore.doc_status.get_by_id(doc_id)
        existing_chunks_count = (
            existing_doc_status.get("chunks_count", 0) if existing_doc_status else 0
        )

        for i, item in enumerate(multimodal_items):
            try:
                content_type = item.get("type", "unknown")
                self.logger.info(
                    f"Processing item {i+1}/{len(multimodal_items)}: {content_type} content"
                )

                # Select appropriate processor
                processor = get_processor_for_type(self.modal_processors, content_type)

                if processor:
                    # Prepare item info for context extraction
                    item_info = {
                        "page_idx": item.get("page_idx", 0),
                        "index": i,
                        "type": content_type,
                    }

                    # Process content and get chunk results instead of immediately merging
                    (
                        enhanced_caption,
                        entity_info,
                        chunk_results,
                    ) = await processor.process_multimodal_content(
                        modal_content=item,
                        content_type=content_type,
                        file_path=file_name,
                        item_info=item_info,  # Pass item info for context extraction
                        batch_mode=True,
                        doc_id=doc_id,  # Pass doc_id for proper association
                        chunk_order_index=existing_chunks_count
                        + i,  # Proper order index
                    )

                    # Collect chunk results for batch processing
                    all_chunk_results.extend(chunk_results)

                    # Extract chunk ID from the entity_info (actual chunk_id created by processor)
                    if entity_info and "chunk_id" in entity_info:
                        chunk_id = entity_info["chunk_id"]
                        multimodal_chunk_ids.append(chunk_id)

                    self.logger.info(
                        f"{content_type} processing complete: {entity_info.get('entity_name', 'Unknown')}"
                    )
                else:
                    self.logger.warning(
                        f"No suitable processor found for {content_type} type content"
                    )

            except Exception as e:
                self.logger.error(f"Error processing multimodal content: {str(e)}")
                self.logger.debug("Exception details:", exc_info=True)
                continue

        # Update doc_status to include multimodal chunks in the standard chunks_list
        if multimodal_chunk_ids:
            try:
                # Get current document status
                current_doc_status = await self.graphcore.doc_status.get_by_id(doc_id)

                if current_doc_status:
                    existing_chunks_list = current_doc_status.get("chunks_list", [])
                    existing_chunks_count = current_doc_status.get("chunks_count", 0)

                    # Add multimodal chunks to the standard chunks_list
                    updated_chunks_list = existing_chunks_list + multimodal_chunk_ids
                    updated_chunks_count = existing_chunks_count + len(
                        multimodal_chunk_ids
                    )

                    # Update document status with integrated chunk list
                    await self.graphcore.doc_status.upsert(
                        {
                            doc_id: {
                                **current_doc_status,  # Keep existing fields
                                "chunks_list": updated_chunks_list,  # Integrated chunks list
                                "chunks_count": updated_chunks_count,  # Updated total count
                                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
                            }
                        }
                    )

                    # Ensure doc_status update is persisted to disk
                    await self.graphcore.doc_status.index_done_callback()

                    self.logger.info(
                        f"Updated doc_status with {len(multimodal_chunk_ids)} multimodal chunks integrated into chunks_list"
                    )

            except Exception as e:
                self.logger.warning(
                    f"Error updating doc_status with multimodal chunks: {e}"
                )

        # Batch merge all multimodal content results (similar to text content processing)
        if all_chunk_results:
            from graphcore.coregraph.operate import merge_nodes_and_edges
            from graphcore.coregraph.kg.shared_storage import (
                get_namespace_data,
                get_namespace_lock,
            )

            workspace = getattr(self.graphcore, "workspace", None)
            # Get pipeline status and lock from shared storage
            pipeline_status = await get_namespace_data(
                "pipeline_status", workspace=workspace
            )
            pipeline_status_lock = get_namespace_lock(
                "pipeline_status", workspace=workspace
            )

            await merge_nodes_and_edges(
                chunk_results=all_chunk_results,
                knowledge_graph_inst=self.graphcore.chunk_entity_relation_graph,
                entity_vdb=self.graphcore.entities_vdb,
                relationships_vdb=self.graphcore.relationships_vdb,
                global_config=self.graphcore.__dict__,
                full_entities_storage=self.graphcore.full_entities,
                full_relations_storage=self.graphcore.full_relations,
                doc_id=doc_id,
                pipeline_status=pipeline_status,
                pipeline_status_lock=pipeline_status_lock,
                llm_response_cache=self.graphcore.llm_response_cache,
                current_file_number=1,
                total_files=1,
                file_path=file_name,
            )

            await self.graphcore._insert_done()

        self.logger.info("Individual multimodal content processing complete")

        # Mark multimodal content as processed
        await self._mark_multimodal_processing_complete(doc_id)

    async def _process_multimodal_content_batch_type_aware(
        self, multimodal_items: List[Dict[str, Any]], file_path: str, doc_id: str
    ):
        """
        Type-aware batch processing that selects correct processors based on content type.
        This is the corrected implementation that handles different modality types properly.

        Args:
            multimodal_items: List of multimodal items with different types
            file_path: File path for citation
            doc_id: Document ID for proper association
        """
        if not multimodal_items:
            self.logger.debug("No multimodal content to process")
            return

        # Get existing chunks count for proper order indexing
        try:
            existing_doc_status = await self.graphcore.doc_status.get_by_id(doc_id)
            existing_chunks_count = (
                existing_doc_status.get("chunks_count", 0) if existing_doc_status else 0
            )
        except Exception:
            existing_chunks_count = 0

        # Use GraphCore's concurrency control
        semaphore = asyncio.Semaphore(getattr(self.graphcore, "max_parallel_insert", 2))

        # Progress tracking variables
        total_items = len(multimodal_items)
        completed_count = 0
        progress_lock = asyncio.Lock()

        # Log processing start
        self.logger.info(f"Starting to process {total_items} multimodal content items")

        # Stage 1: Concurrent generation of descriptions using correct processors for each type
        async def process_single_item_with_correct_processor(
            item: Dict[str, Any], index: int, file_path: str
        ):
            nonlocal completed_count
            async with semaphore:
                attempts = 3
                last_error = None
                for attempt in range(attempts):
                    try:
                        content_type = item.get("type", "unknown")
                        processor = get_processor_for_type(
                            self.modal_processors, content_type
                        )
                        if not processor:
                            self.logger.warning(
                                f"No processor found for type: {content_type}"
                            )
                            return None
                        item_info = {
                            "page_idx": item.get("page_idx", 0),
                            "index": index,
                            "type": content_type,
                        }
                        (
                            description,
                            entity_info,
                        ) = await processor.generate_description_only(
                            modal_content=item,
                            content_type=content_type,
                            item_info=item_info,
                            entity_name=None,
                        )
                        async with progress_lock:
                            completed_count += 1
                            if (
                                completed_count % max(1, total_items // 10) == 0
                                or completed_count == total_items
                            ):
                                progress_percent = (completed_count / total_items) * 100
                                self.logger.info(
                                    f"Multimodal chunk generation progress: {completed_count}/{total_items} ({progress_percent:.1f}%)"
                                )
                        return {
                            "index": index,
                            "content_type": content_type,
                            "description": description,
                            "entity_info": entity_info,
                            "original_item": item,
                            "item_info": item_info,
                            "chunk_order_index": existing_chunks_count + index,
                            "processor": processor,
                            "file_path": file_path,
                        }
                    except Exception as e:
                        last_error = e
                        if attempt < attempts - 1:
                            await asyncio.sleep(1 * (2 ** attempt))
                            continue
                        async with progress_lock:
                            completed_count += 1
                            if (
                                completed_count % max(1, total_items // 10) == 0
                                or completed_count == total_items
                            ):
                                progress_percent = (completed_count / total_items) * 100
                                self.logger.info(
                                    f"Multimodal chunk generation progress: {completed_count}/{total_items} ({progress_percent:.1f}%)"
                                )
                        self.logger.error(
                            f"Error generating description for {item.get('type','unknown')} item {index}: {last_error}"
                        )
                        return None

        # Process all items concurrently with correct processors
        tasks = [
            asyncio.create_task(
                process_single_item_with_correct_processor(item, i, file_path)
            )
            for i, item in enumerate(multimodal_items)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter successful results
        multimodal_data_list = []
        for result in results:
            if isinstance(result, Exception):
                self.logger.error(f"Task failed: {result}")
                continue
            if result is not None:
                multimodal_data_list.append(result)

        if not multimodal_data_list:
            self.logger.warning("No valid multimodal descriptions generated")
            return

        self.logger.info(
            f"Generated descriptions for {len(multimodal_data_list)}/{len(multimodal_items)} multimodal items using correct processors"
        )

        # Stage 2: Convert to CoreGraph chunks format
        coregraph_chunks = self._convert_to_coregraph_chunks_type_aware(
            multimodal_data_list, file_path, doc_id
        )

        # Stage 3: Store chunks to CoreGraph storage
        await self._store_chunks_to_coregraph_storage_type_aware(coregraph_chunks)

        # Stage 3.5: Store multimodal main entities to entities_vdb and full_entities
        await self._store_multimodal_main_entities(
            multimodal_data_list, coregraph_chunks, file_path, doc_id
        )

        # Track chunk IDs for doc_status update
        chunk_ids = list(coregraph_chunks.keys())

        extraction_mode = (self.config.relation_extraction_mode or "coregraph").lower()
        if extraction_mode not in {"coregraph", "hypergraph"}:
            self.logger.warning(
                "Unknown relation_extraction_mode '%s', falling back to CoreGraph.",
                extraction_mode,
            )
            extraction_mode = "coregraph"

        chunk_results: List[Tuple] | None = None
        if extraction_mode == "hypergraph":
            await self._batch_extract_entities_hypergraph_prompt(
                coregraph_chunks, doc_id=doc_id, file_path=file_path
            )
        else:
            # Stage 4: Use CoreGraph's batch entity relation extraction
            chunk_results = await self._batch_extract_entities_coregraph_style_type_aware(
                coregraph_chunks
            )

        if chunk_results is not None:
            # Stage 5: Add belongs_to relations (multimodal-specific)
            enhanced_chunk_results = (
                await self._batch_add_belongs_to_relations_type_aware(
                    chunk_results, multimodal_data_list
                )
            )

            # Stage 6: Use CoreGraph's batch merge
            await self._batch_merge_coregraph_style_type_aware(
                enhanced_chunk_results, file_path, doc_id
            )
        else:
            self.logger.info(
                "HyperGraph prompt-based extraction selected; skipping CoreGraph merge pipeline."
            )

        # Stage 7: Update doc_status with integrated chunks_list
        await self._update_doc_status_with_chunks_type_aware(doc_id, chunk_ids)

    def _convert_to_coregraph_chunks_type_aware(
        self, multimodal_data_list: List[Dict[str, Any]], file_path: str, doc_id: str
    ) -> Dict[str, Any]:
        """Convert multimodal data to CoreGraph standard chunks format"""

        chunks = {}

        for data in multimodal_data_list:
            description = data["description"]
            entity_info = data["entity_info"]
            chunk_order_index = data["chunk_order_index"]
            content_type = data["content_type"]
            original_item = data["original_item"]

            # Apply the appropriate chunk template based on content type
            formatted_chunk_content = self._apply_chunk_template(
                content_type, original_item, description
            )

            # Generate chunk_id
            chunk_id = compute_mdhash_id(formatted_chunk_content, prefix="chunk-")

            # Calculate tokens
            tokens = len(self.graphcore.tokenizer.encode(formatted_chunk_content))

            # Build CoreGraph standard chunk format
            chunks[chunk_id] = {
                "content": formatted_chunk_content,  # Now uses the templated content
                "tokens": tokens,
                "full_doc_id": doc_id,
                "chunk_order_index": chunk_order_index,
                "file_path": os.path.basename(file_path),
                "llm_cache_list": [],  # CoreGraph will populate this field
                # Multimodal-specific metadata
                "is_multimodal": True,
                "modal_entity_name": entity_info["entity_name"],
                "original_type": data["content_type"],
                "page_idx": data["item_info"].get("page_idx", 0),
            }

        self.logger.debug(
            f"Converted {len(chunks)} multimodal items to multimodal chunks format"
        )
        return chunks

    def _apply_chunk_template(
        self, content_type: str, original_item: Dict[str, Any], description: str
    ) -> str:
        """
        Apply the appropriate chunk template based on content type

        Args:
            content_type: Type of content (image, table, equation, generic)
            original_item: Original multimodal item data
            description: Enhanced description generated by the processor

        Returns:
            Formatted chunk content using the appropriate template
        """
        from docthinker.prompt import PROMPTS

        try:
            if content_type == "image":
                image_path = original_item.get("img_path", "")
                captions = original_item.get(
                    "image_caption", original_item.get("img_caption", [])
                )
                footnotes = original_item.get(
                    "image_footnote", original_item.get("img_footnote", [])
                )

                return PROMPTS["image_chunk"].format(
                    image_path=image_path,
                    captions=", ".join(captions) if captions else "None",
                    footnotes=", ".join(footnotes) if footnotes else "None",
                    enhanced_caption=description,
                )

            elif content_type == "table":
                table_img_path = original_item.get("img_path", "")
                table_caption = original_item.get("table_caption", [])
                table_body = original_item.get("table_body", "")
                table_footnote = original_item.get("table_footnote", [])

                return PROMPTS["table_chunk"].format(
                    table_img_path=table_img_path,
                    table_caption=", ".join(table_caption) if table_caption else "None",
                    table_body=table_body,
                    table_footnote=", ".join(table_footnote)
                    if table_footnote
                    else "None",
                    enhanced_caption=description,
                )

            elif content_type == "equation":
                equation_text = original_item.get("text", "")
                equation_format = original_item.get("text_format", "")

                return PROMPTS["equation_chunk"].format(
                    equation_text=equation_text,
                    equation_format=equation_format,
                    enhanced_caption=description,
                )

            else:  # generic or unknown types
                content = str(original_item.get("content", original_item))

                return PROMPTS["generic_chunk"].format(
                    content_type=content_type.title(),
                    content=content,
                    enhanced_caption=description,
                )

        except Exception as e:
            self.logger.warning(
                f"Error applying chunk template for {content_type}: {e}"
            )
            # Fallback to just the description if template fails
            return description

    async def _store_chunks_to_graphcore.coregraph_storage_type_aware(
        self, chunks: Dict[str, Any]
    ):
        """Store chunks to storage"""
        try:
            # Store in text_chunks storage (required for extract_entities)
            await self.graphcore.text_chunks.upsert(chunks)

            # Store in chunks vector database for retrieval
            await self.graphcore.chunks_vdb.upsert(chunks)

            self.logger.debug(f"Stored {len(chunks)} multimodal chunks to storage")

        except Exception as e:
            self.logger.error(f"Error storing chunks to storage: {e}")
            raise

    async def _store_multimodal_main_entities(
        self,
        multimodal_data_list: List[Dict[str, Any]],
        graphcore.coregraph_chunks: Dict[str, Any],
        file_path: str,
        doc_id: str = None,
    ):
        """
        Store multimodal main entities to entities_vdb and full_entities.
        This ensures that entities like "TableName (table)" are properly indexed.

        Args:
            multimodal_data_list: List of processed multimodal data with entity info
            graphcore.coregraph_chunks: Chunks in GraphCore format (already formatted with templates)
            file_path: File path for the entities
            doc_id: Document ID for full_entities storage
        """
        if not multimodal_data_list:
            return

        # Create entities_vdb entries for all multimodal main entities
        entities_to_store = {}

        for data in multimodal_data_list:
            entity_info = data["entity_info"]
            entity_name = entity_info["entity_name"]
            description = data["description"]
            content_type = data["content_type"]
            original_item = data["original_item"]

            # Apply the same chunk template to get the formatted content
            formatted_chunk_content = self._apply_chunk_template(
                content_type, original_item, description
            )

            # Generate chunk_id using the formatted content (same as in _convert_to_graphcore.coregraph_chunks)
            chunk_id = compute_mdhash_id(formatted_chunk_content, prefix="chunk-")

            # Generate entity_id using GraphCore's standard format
            entity_id = compute_mdhash_id(entity_name, prefix="ent-")

            # Create entity data in GraphCore format
            entity_data = {
                "entity_name": entity_name,
                "entity_type": entity_info.get("entity_type", content_type),
                "content": entity_info.get("summary", description),
                "source_id": chunk_id,
                "file_path": os.path.basename(file_path),
            }

            entities_to_store[entity_id] = entity_data

        if entities_to_store:
            try:
                # Store entities in knowledge graph
                for entity_id, entity_data in entities_to_store.items():
                    entity_name = entity_data["entity_name"]

                    # Create node data for knowledge graph
                    node_data = {
                        "entity_id": entity_name,
                        "entity_type": entity_data["entity_type"],
                        "description": entity_data["content"],
                        "source_id": entity_data["source_id"],
                        "file_path": entity_data["file_path"],
                        "created_at": int(time.time()),
                    }

                    # Store in knowledge graph
                    await self.graphcore.chunk_entity_relation_graph.upsert_node(
                        entity_name, node_data
                    )

                # Store in entities_vdb
                await self.graphcore.entities_vdb.upsert(entities_to_store)
                await self.graphcore.entities_vdb.index_done_callback()

                # NEW: Store multimodal main entities in full_entities storage
                if doc_id and self.graphcore.full_entities:
                    await self._store_multimodal_entities_to_full_entities(
                        entities_to_store, doc_id
                    )

                self.logger.debug(
                    f"Stored {len(entities_to_store)} multimodal main entities to knowledge graph, entities_vdb, and full_entities"
                )

            except Exception as e:
                self.logger.error(f"Error storing multimodal main entities: {e}")
                raise

    async def _store_multimodal_entities_to_full_entities(
        self, entities_to_store: Dict[str, Any], doc_id: str
    ):
        """
        Store multimodal main entities to full_entities storage.

        Args:
            entities_to_store: Dictionary of entities to store
            doc_id: Document ID for grouping entities
        """
        try:
            # Get current full_entities data for this document
            current_doc_entities = await self.graphcore.full_entities.get_by_id(doc_id)

            if current_doc_entities is None:
                # Create new document entry
                entity_names = list(
                    entity_data["entity_name"]
                    for entity_data in entities_to_store.values()
                )
                doc_entities_data = {
                    "entity_names": entity_names,
                    "count": len(entity_names),
                    "update_time": int(time.time()),
                }
            else:
                # Update existing document entry
                existing_entity_names = set(
                    current_doc_entities.get("entity_names", [])
                )
                new_entity_names = [
                    entity_data["entity_name"]
                    for entity_data in entities_to_store.values()
                ]

                # Add new multimodal entities to the list (avoid duplicates)
                for entity_name in new_entity_names:
                    existing_entity_names.add(entity_name)

                doc_entities_data = {
                    "entity_names": list(existing_entity_names),
                    "count": len(existing_entity_names),
                    "update_time": int(time.time()),
                }

            # Store updated data
            await self.graphcore.full_entities.upsert({doc_id: doc_entities_data})
            await self.graphcore.full_entities.index_done_callback()

            self.logger.debug(
                f"Added {len(entities_to_store)} multimodal main entities to full_entities for doc {doc_id}"
            )

        except Exception as e:
            self.logger.error(
                f"Error storing multimodal entities to full_entities: {e}"
            )
            raise

    async def _batch_extract_entities_graphcore.coregraph_style_type_aware(
        self, graphcore.coregraph_chunks: Dict[str, Any]
    ) -> List[Tuple]:
        """Use GraphCore's extract_entities for batch entity relation extraction"""
        from graphcore.coregraph.kg.shared_storage import (
            get_namespace_data,
            get_namespace_lock,
        )
        from graphcore.coregraph.operate import extract_entities

        workspace = getattr(self.graphcore, "workspace", None)
        # Get pipeline status (consistent with GraphCore)
        pipeline_status = await get_namespace_data(
            "pipeline_status", workspace=workspace
        )
        pipeline_status_lock = get_namespace_lock(
            "pipeline_status", workspace=workspace
        )

        # Retry per-chunk extraction to avoid failing the whole batch (execute across chunks concurrently)
        all_results: List[Tuple] = []
        concurrency = max(1, int(getattr(self.graphcore, "llm_model_max_async", 8)))
        semaphore = asyncio.Semaphore(concurrency)

        async def _extract_one(chunk_id: str, chunk: Dict[str, Any]) -> List[Tuple]:
            async with semaphore:
                single_chunk = {chunk_id: chunk}
                attempts = 3
                last_error = None
                for attempt in range(attempts):
                    try:
                        from graphcore.coregraph.llm.openai import openai_complete_if_cache
                        # Build a dedicated 8B llm_model_func for extraction-only, without affecting other paths
                        async def _llm_8b(prompt, system_prompt=None, history_messages=[], **kwargs):
                            api_key = (
                                getattr(self.config, "bltcy_api_key", None)
                                or os.getenv("BLTCY_API_KEY")
                                or os.getenv("LLM_BINDING_API_KEY")
                            )
                            base_url = (
                                getattr(self.config, "bltcy_api_base", None)
                                or os.getenv("BLTCY_API_BASE")
                                or os.getenv("LLM_BINDING_HOST", "https://dashscope.aliyuncs.com/compatible-mode/v1")
                            )
                            return await openai_complete_if_cache(
                                "qwen3-8b",
                                prompt,
                                system_prompt=system_prompt,
                                history_messages=history_messages,
                                api_key=api_key,
                                base_url=base_url,
                                timeout=360,
                                extra_body={"enable_thinking": False},
                                **kwargs,
                            )
                        # Use a shallow copy of GraphCore config and override only llm_model_func
                        alt_global_cfg = dict(self.graphcore.__dict__)
                        alt_global_cfg["llm_model_func"] = _llm_8b
                        result = await extract_entities(
                            chunks=single_chunk,
                            global_config=alt_global_cfg,
                            pipeline_status=pipeline_status,
                            pipeline_status_lock=pipeline_status_lock,
                            llm_response_cache=self.graphcore.llm_response_cache,
                            text_chunks_storage=self.graphcore.text_chunks,
                        )
                        return result or []
                    except Exception as e:
                        last_error = e
                        if attempt < attempts - 1:
                            await asyncio.sleep(1 * (2 ** attempt))
                            continue
                        self.logger.error(
                            f"Extraction failed for chunk {chunk_id}: {last_error}"
                        )
                        return []

        tasks = [
            asyncio.create_task(_extract_one(chunk_id, chunk))
            for chunk_id, chunk in graphcore.coregraph_chunks.items()
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for res in results:
            if isinstance(res, Exception):
                self.logger.error(f"Extraction task failed: {res}")
                continue
            if res:
                all_results.extend(res)

        self.logger.info(
            f"Extracted entities from {len(all_results)} results across {len(graphcore.coregraph_chunks)} multimodal chunks"
        )
        
        # Add entities and relationships to knowledge base
        if hasattr(self, 'add_knowledge_entry'):
            for result in all_results:
                # Each result is a tuple containing entities and relationships
                chunk_id = result[0] if result else None
                if chunk_id and chunk_id in graphcore.coregraph_chunks:
                    chunk = graphcore.coregraph_chunks[chunk_id]
                    doc_id = chunk.get('full_doc_id')
                    content = chunk.get('content', '')
                    
                    # Add chunk content to knowledge base
                    chunk_entry_id = self.add_knowledge_entry(
                        content=content,
                        entry_type='chunk',
                        metadata={'chunk_id': chunk_id, 'doc_id': doc_id, 'is_multimodal': chunk.get('is_multimodal', False)},
                        kb_name=f'doc_{doc_id}' if doc_id else 'global'
                    )
                    
                    # If entities were extracted, add them to knowledge base and link to chunk
                    if len(result) > 1 and result[1]:
                        entities = result[1]
                        for entity in entities:
                            entity_name = entity.get('entity_name', '')
                            entity_type = entity.get('entity_type', 'unknown')
                            entity_content = entity.get('content', '')
                            
                            # Add entity to knowledge base
                            entity_entry_id = self.add_knowledge_entry(
                                content=entity_content,
                                entry_type='entity',
                                metadata={'entity_name': entity_name, 'entity_type': entity_type, 'chunk_id': chunk_id, 'doc_id': doc_id},
                                kb_name=f'doc_{doc_id}' if doc_id else 'global'
                            )
                            
                            # Link entity to chunk entry
                            if hasattr(self, 'knowledge_base_manager') and self.knowledge_base_manager:
                                kb_name = f'doc_{doc_id}' if doc_id else 'global'
                                # Add relation between chunk and entity
                                chunk_entry = self.knowledge_base_manager.get_knowledge_base(kb_name).get_entry(chunk_entry_id)
                                if chunk_entry:
                                    chunk_entry.add_relation('contains_entity', entity_entry_id)
                                
                                entity_entry = self.knowledge_base_manager.get_knowledge_base(kb_name).get_entry(entity_entry_id)
                                if entity_entry:
                                    entity_entry.add_relation('contained_in', chunk_entry_id)
                                
                                # Save changes to knowledge base
                                self.knowledge_base_manager.get_knowledge_base(kb_name).save(str(Path(self.working_dir) / 'knowledge_bases' / f'{kb_name}.json'))
                    
                    # If relationships were extracted, add them to knowledge base and link to entities
                    if len(result) > 2 and result[2]:
                        relationships = result[2]
                        for relation in relationships:
                            source_entity = relation.get('source_entity', '')
                            target_entity = relation.get('target_entity', '')
                            relation_type = relation.get('relation_type', '')
                            relation_content = relation.get('content', '')
                            
                            # Add relationship to knowledge base
                            relation_entry_id = self.add_knowledge_entry(
                                content=relation_content,
                                entry_type='relationship',
                                metadata={'source_entity': source_entity, 'target_entity': target_entity, 'relation_type': relation_type, 'chunk_id': chunk_id, 'doc_id': doc_id},
                                kb_name=f'doc_{doc_id}' if doc_id else 'global'
                            )
        
        # Add entities and relationships to knowledge base
        if hasattr(self, 'add_knowledge_entry'):
            for result in all_results:
                if result and len(result) > 0:
                    chunk_id = result[0] if isinstance(result[0], str) else None
                    if chunk_id and chunk_id in graphcore.coregraph_chunks:
                        chunk = graphcore.coregraph_chunks[chunk_id]
                        # Add chunk content to knowledge base
                        self.add_knowledge_entry(
                            content=chunk['content'],
                            entry_type='chunk',
                            metadata={
                                'chunk_id': chunk_id,
                                'doc_id': chunk['full_doc_id'],
                                'file_path': chunk['file_path'],
                                'is_multimodal': chunk.get('is_multimodal', False)
                            }
                        )
        
        return all_results

    async def _batch_extract_entities_hypergraph_prompt(
        self,
        graphcore.coregraph_chunks: Dict[str, Any],
        *,
        doc_id: str,
        file_path: str,
    ) -> None:
        """Use HyperGraphRAG prompt pipeline for entity + relation extraction."""
        from docthinker.hypergraph.operate import extract_entities as hyper_extract_entities

        if self.graphcore is None:
            raise RuntimeError("GraphCore must be initialized before extraction.")

        llm_func = self.llm_model_func or getattr(self.graphcore, "llm_model_func", None)
        if llm_func is None:
            raise RuntimeError(
                "HyperGraph prompt extraction requires an llm_model_func instance."
            )

        hyper_chunks: Dict[str, Dict[str, Any]] = {}
        for chunk_id, chunk in graphcore.coregraph_chunks.items():
            content = chunk.get("content", "")
            metadata = {
                "doc_id": doc_id,
                "source_path": chunk.get("file_path", os.path.basename(file_path)),
                "type": chunk.get("original_type", "text"),
            }
            hyper_chunks[chunk_id] = {"content": content, "metadata": metadata}

        addon_params = dict(getattr(self.graphcore, "addon_params", {}) or {})
        language = (self.config.hyper_prompt_language or "").strip()
        if language:
            addon_params["language"] = language

        entity_types = [
            item.strip()
            for item in (self.config.hyper_prompt_entity_types or [])
            if item.strip()
        ]
        if entity_types:
            addon_params["entity_types"] = entity_types

        addon_params.setdefault(
            "example_number", self.config.hyper_prompt_example_number
        )

        global_config = {
            "llm_model_func": llm_func,
            "llm_model_max_token_size": getattr(
                self.graphcore, "llm_model_max_token_size", 4096
            ),
            "tiktoken_model_name": getattr(
                self.graphcore, "tiktoken_model_name", "gpt-4o-mini"
            ),
            "entity_summary_to_max_tokens": getattr(
                self.graphcore, "entity_summary_to_max_tokens", 500
            ),
            "entity_extract_max_gleaning": self.config.hyper_prompt_max_gleaning,
            "addon_params": addon_params,
        }

        await hyper_extract_entities(
            chunks=hyper_chunks,
            knowledge_graph_inst=self.graphcore.chunk_entity_relation_graph,
            entity_vdb=self.graphcore.entities_vdb,
            hyperedge_vdb=self.graphcore.relationships_vdb,
            global_config=global_config,
        )
        self.logger.info(
            "Completed HyperGraph prompt-based relation extraction for %s",
            doc_id,
        )

    async def _batch_add_belongs_to_relations_type_aware(
        self, chunk_results: List[Tuple], multimodal_data_list: List[Dict[str, Any]]
    ) -> List[Tuple]:
        """Add belongs_to relations for multimodal entities"""
        # Create mapping from chunk_id to modal_entity_name
        chunk_to_modal_entity = {}
        chunk_to_file_path = {}

        for data in multimodal_data_list:
            description = data["description"]
            content_type = data["content_type"]
            original_item = data["original_item"]

            # Use the same template formatting as in _convert_to_graphcore.coregraph_chunks_type_aware
            formatted_chunk_content = self._apply_chunk_template(
                content_type, original_item, description
            )
            chunk_id = compute_mdhash_id(formatted_chunk_content, prefix="chunk-")

            chunk_to_modal_entity[chunk_id] = data["entity_info"]["entity_name"]
            chunk_to_file_path[chunk_id] = data.get("file_path", "multimodal_content")

        enhanced_chunk_results = []
        belongs_to_count = 0

        for maybe_nodes, maybe_edges in chunk_results:
            # Find corresponding modal_entity_name for this chunk
            chunk_id = None
            for nodes_dict in maybe_nodes.values():
                if nodes_dict:
                    chunk_id = nodes_dict[0].get("source_id")
                    break

            if chunk_id and chunk_id in chunk_to_modal_entity:
                modal_entity_name = chunk_to_modal_entity[chunk_id]
                file_path = chunk_to_file_path.get(chunk_id, "multimodal_content")

                # Add belongs_to relations for all extracted entities
                for entity_name in maybe_nodes.keys():
                    if entity_name != modal_entity_name:  # Avoid self-relation
                        belongs_to_relation = {
                            "src_id": entity_name,
                            "tgt_id": modal_entity_name,
                            "description": f"Entity {entity_name} belongs to {modal_entity_name}",
                            "keywords": "belongs_to,part_of,contained_in",
                            "source_id": chunk_id,
                            "weight": 10.0,
                            "file_path": file_path,
                        }

                        # Add to maybe_edges
                        edge_key = (entity_name, modal_entity_name)
                        if edge_key not in maybe_edges:
                            maybe_edges[edge_key] = []
                        maybe_edges[edge_key].append(belongs_to_relation)
                        belongs_to_count += 1

            enhanced_chunk_results.append((maybe_nodes, maybe_edges))

        self.logger.info(
            f"Added {belongs_to_count} belongs_to relations for multimodal entities"
        )
        return enhanced_chunk_results

    async def _batch_merge_graphcore.coregraph_style_type_aware(
        self, enhanced_chunk_results: List[Tuple], file_path: str, doc_id: str = None
    ):
        """Use GraphCore's merge_nodes_and_edges for batch merge"""
        from graphcore.coregraph.kg.shared_storage import (
            get_namespace_data,
            get_namespace_lock,
        )
        from graphcore.coregraph.operate import merge_nodes_and_edges

        workspace = getattr(self.graphcore, "workspace", None)
        pipeline_status = await get_namespace_data(
            "pipeline_status", workspace=workspace
        )
        pipeline_status_lock = get_namespace_lock(
            "pipeline_status", workspace=workspace
        )

        await merge_nodes_and_edges(
            chunk_results=enhanced_chunk_results,
            knowledge_graph_inst=self.graphcore.chunk_entity_relation_graph,
            entity_vdb=self.graphcore.entities_vdb,
            relationships_vdb=self.graphcore.relationships_vdb,
            global_config=self.graphcore.__dict__,
            full_entities_storage=self.graphcore.full_entities,
            full_relations_storage=self.graphcore.full_relations,
            doc_id=doc_id,
            pipeline_status=pipeline_status,
            pipeline_status_lock=pipeline_status_lock,
            llm_response_cache=self.graphcore.llm_response_cache,
            current_file_number=1,
            total_files=1,
            file_path=os.path.basename(file_path),
        )

        await self.graphcore._insert_done()

    async def _update_doc_status_with_chunks_type_aware(
        self, doc_id: str, chunk_ids: List[str]
    ):
        """Update document status with multimodal chunks"""
        try:
            # Get current document status
            current_doc_status = await self.graphcore.doc_status.get_by_id(doc_id)

            if current_doc_status:
                existing_chunks_list = current_doc_status.get("chunks_list", [])
                existing_chunks_count = current_doc_status.get("chunks_count", 0)

                # Add multimodal chunks to the standard chunks_list
                updated_chunks_list = existing_chunks_list + chunk_ids
                updated_chunks_count = existing_chunks_count + len(chunk_ids)

                # Update document status with integrated chunk list
                await self.graphcore.doc_status.upsert(
                    {
                        doc_id: {
                            **current_doc_status,  # Keep existing fields
                            "chunks_list": updated_chunks_list,  # Integrated chunks list
                            "chunks_count": updated_chunks_count,  # Updated total count
                            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
                        }
                    }
                )

                # Ensure doc_status update is persisted to disk
                await self.graphcore.doc_status.index_done_callback()

                self.logger.info(
                    f"Updated doc_status: added {len(chunk_ids)} multimodal chunks to standard chunks_list "
                    f"(total chunks: {updated_chunks_count})"
                )

        except Exception as e:
            self.logger.warning(
                f"Error updating doc_status with multimodal chunks: {e}"
            )

    async def _mark_multimodal_processing_complete(self, doc_id: str):
        """Mark multimodal content processing as complete in the document status."""
        try:
            current_doc_status = await self.graphcore.doc_status.get_by_id(doc_id)
            if current_doc_status:
                await self.graphcore.doc_status.upsert(
                    {
                        doc_id: {
                            **current_doc_status,
                            "multimodal_processed": True,
                            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
                        }
                    }
                )
                await self.graphcore.doc_status.index_done_callback()
                self.logger.debug(
                    f"Marked multimodal content processing as complete for document {doc_id}"
                )
        except Exception as e:
            self.logger.warning(
                f"Error marking multimodal processing as complete for document {doc_id}: {e}"
            )

    async def is_document_fully_processed(self, doc_id: str) -> bool:
        """
        Check if a document is fully processed (both text and multimodal content).

        Args:
            doc_id: Document ID to check

        Returns:
            bool: True if both text and multimodal content are processed
        """
        try:
            doc_status = await self.graphcore.doc_status.get_by_id(doc_id)
            if not doc_status:
                return False

            text_processed = doc_status.get("status") == DocStatus.PROCESSED
            multimodal_processed = doc_status.get("multimodal_processed", False)

            return text_processed and multimodal_processed

        except Exception as e:
            self.logger.error(
                f"Error checking document processing status for {doc_id}: {e}"
            )
            return False

    async def get_document_processing_status(self, doc_id: str) -> Dict[str, Any]:
        """
        Get detailed processing status for a document.

        Args:
            doc_id: Document ID to check

        Returns:
            Dict with processing status details
        """
        try:
            doc_status = await self.graphcore.doc_status.get_by_id(doc_id)
            if not doc_status:
                return {
                    "exists": False,
                    "text_processed": False,
                    "multimodal_processed": False,
                    "fully_processed": False,
                    "chunks_count": 0,
                }

            text_processed = doc_status.get("status") == DocStatus.PROCESSED
            multimodal_processed = doc_status.get("multimodal_processed", False)
            fully_processed = text_processed and multimodal_processed

            return {
                "exists": True,
                "text_processed": text_processed,
                "multimodal_processed": multimodal_processed,
                "fully_processed": fully_processed,
                "chunks_count": doc_status.get("chunks_count", 0),
                "chunks_list": doc_status.get("chunks_list", []),
                "status": doc_status.get("status", ""),
                "updated_at": doc_status.get("updated_at", ""),
                "raw_status": doc_status,
            }

        except Exception as e:
            self.logger.error(
                f"Error getting document processing status for {doc_id}: {e}"
            )
            return {
                "exists": False,
                "error": str(e),
                "text_processed": False,
                "multimodal_processed": False,
                "fully_processed": False,
                "chunks_count": 0,
            }

    async def _emit_hyper_chunks(
        self,
        doc_id: str,
        *,
        file_path: str,
        text_content: str,
        multimodal_items: List[Dict[str, Any]],
    ) -> None:
        """Notify optional hyper chunk sink for auto-thinking integration."""
        sink = getattr(self, "hyper_chunk_sink", None)
        if sink is None:
            return

        try:
            result = sink(
                doc_id,
                text_content=text_content,
                multimodal_items=multimodal_items,
                file_path=file_path,
            )
            if asyncio.iscoroutine(result):
                await result
        except Exception as exc:  # noqa: BLE001
            self.logger.warning(f"hyper_chunk_sink failed for {doc_id}: {exc}")

    async def process_document_complete(
        self,
        file_path: str,
        output_dir: str = None,
        parse_method: str = None,
        display_stats: bool = None,
        split_by_character: str | None = None,
        split_by_character_only: bool = False,
        doc_id: str | None = None,
        **kwargs,
    ):
        """
        Complete document processing workflow

        Args:
            file_path: Path to the file to process
            output_dir: output directory (defaults to config.parser_output_dir)
            parse_method: Parse method (defaults to config.parse_method)
            display_stats: Whether to display content statistics (defaults to config.display_content_stats)
            split_by_character: Optional character to split the text by
            split_by_character_only: If True, split only by the specified character
            doc_id: Optional document ID, if not provided will be generated from content
            **kwargs: Additional parameters for parser (e.g., lang, device, start_page, end_page, formula, table, backend, source)
        """
        # Ensure GraphCore is initialized
        await self._ensure_graphcore.coregraph_initialized()

        # Use config defaults if not provided
        if output_dir is None:
            output_dir = self.config.parser_output_dir
        if parse_method is None:
            parse_method = self.config.parse_method
        if display_stats is None:
            display_stats = self.config.display_content_stats

        self.logger.info(f"Starting complete document processing: {file_path}")

        # Step 1: Parse document
        content_list, content_based_doc_id = await self.parse_document(
            file_path, output_dir, parse_method, display_stats, **kwargs
        )

        # Use provided doc_id or fall back to content-based doc_id
        if doc_id is None:
            doc_id = content_based_doc_id

        # Step 2: Separate text and multimodal content
        text_content, multimodal_items = separate_content(content_list)

        # Notify auto-thinking orchestrator if HyperGraph extraction is enabled
        if getattr(self.config, "enable_hyper_entity_extraction", True):
            await self._emit_hyper_chunks(
                doc_id,
                file_path=file_path,
                text_content=text_content,
                multimodal_items=multimodal_items,
            )

        # Step 2.5: Set content source for context extraction in multimodal processing
        if hasattr(self, "set_content_source_for_context") and multimodal_items:
            self.logger.info(
                "Setting content source for context-aware multimodal processing..."
            )
            self.set_content_source_for_context(
                content_list, self.config.content_format
            )

        # Step 3: Insert pure text content with all parameters
        if text_content.strip():
            file_name = os.path.basename(file_path)
            await insert_text_content(
                self.graphcore,
                input=text_content,
                file_paths=file_name,
                split_by_character=split_by_character,
                split_by_character_only=split_by_character_only,
                ids=doc_id,
            )

        # Step 4: Process multimodal content (using specialized processors)
        if multimodal_items:
            await self._process_multimodal_content(multimodal_items, file_path, doc_id)
        else:
            # If no multimodal content, mark multimodal processing as complete
            # This ensures the document status properly reflects completion of all processing
            await self._mark_multimodal_processing_complete(doc_id)
            self.logger.debug(
                f"No multimodal content found in document {doc_id}, marked multimodal processing as complete"
            )

        self.logger.info(f"Document {file_path} processing complete!")

    async def process_document_complete_graphcore.coregraph_api(
        self,
        file_path: str,
        output_dir: str = None,
        parse_method: str = None,
        display_stats: bool = None,
        split_by_character: str | None = None,
        split_by_character_only: bool = False,
        doc_id: str | None = None,
        scheme_name: str | None = None,
        parser: str | None = None,
        **kwargs,
    ):
        """
        API exclusively for GraphCore calls: Complete document processing workflow

        Args:
            file_path: Path to the file to process
            output_dir: output directory (defaults to config.parser_output_dir)
            parse_method: Parse method (defaults to config.parse_method)
            display_stats: Whether to display content statistics (defaults to config.display_content_stats)
            split_by_character: Optional character to split the text by
            split_by_character_only: If True, split only by the specified character
            doc_id: Optional document ID, if not provided will be generated from content
            **kwargs: Additional parameters for parser (e.g., lang, device, start_page, end_page, formula, table, backend, source)
        """
        file_name = os.path.basename(file_path)
        doc_pre_id = f"doc-pre-{file_name}"
        pipeline_status = None
        pipeline_status_lock = None

        if parser:
            self.config.parser = parser

        current_doc_status = await self.graphcore.doc_status.get_by_id(doc_pre_id)

        try:
            # Ensure GraphCore is initialized
            result = await self._ensure_graphcore.coregraph_initialized()
            if not result["success"]:
                await self.graphcore.doc_status.upsert(
                    {
                        doc_pre_id: {
                            **current_doc_status,
                            "status": DocStatus.FAILED,
                            "error_msg": result["error"],
                        }
                    }
                )
                return False

            # Use config defaults if not provided
            if output_dir is None:
                output_dir = self.config.parser_output_dir
            if parse_method is None:
                parse_method = self.config.parse_method
            if display_stats is None:
                display_stats = self.config.display_content_stats

            self.logger.info(f"Starting complete document processing: {file_path}")

            # Initialize doc status
            current_doc_status = await self.graphcore.doc_status.get_by_id(doc_pre_id)
            if not current_doc_status:
                await self.graphcore.doc_status.upsert(
                    {
                        doc_pre_id: {
                            "status": DocStatus.READY,
                            "content": "",
                            "error_msg": "",
                            "content_summary": "",
                            "multimodal_content": [],
                            "scheme_name": scheme_name,
                            "content_length": 0,
                            "created_at": "",
                            "updated_at": "",
                            "file_path": file_name,
                        }
                    }
                )
                current_doc_status = await self.graphcore.doc_status.get_by_id(
                    doc_pre_id
                )

            from graphcore.coregraph.kg.shared_storage import (
                get_namespace_data,
                get_namespace_lock,
            )

            workspace = getattr(self.graphcore, "workspace", None)
            pipeline_status = await get_namespace_data(
                "pipeline_status", workspace=workspace
            )
            pipeline_status_lock = get_namespace_lock(
                "pipeline_status", workspace=workspace
            )

            # Set processing status
            async with pipeline_status_lock:
                pipeline_status.update({"scan_disabled": True})
                pipeline_status["history_messages"].append("Now is not allowed to scan")

            await self.graphcore.doc_status.upsert(
                {
                    doc_pre_id: {
                        **current_doc_status,
                        "status": DocStatus.HANDLING,
                        "error_msg": "",
                    }
                }
            )

            content_list = []
            content_based_doc_id = ""

            try:
                # Step 1: Parse document
                content_list, content_based_doc_id = await self.parse_document(
                    file_path, output_dir, parse_method, display_stats, **kwargs
                )
            except MineruExecutionError as e:
                error_message = e.error_msg
                if isinstance(e.error_msg, list):
                    error_message = "\n".join(e.error_msg)
                await self.graphcore.doc_status.upsert(
                    {
                        doc_pre_id: {
                            **current_doc_status,
                            "status": DocStatus.FAILED,
                            "error_msg": error_message,
                        }
                    }
                )
                self.logger.info(
                    f"Error processing document {file_path}: MineruExecutionError"
                )
                return False
            except Exception as e:
                await self.graphcore.doc_status.upsert(
                    {
                        doc_pre_id: {
                            **current_doc_status,
                            "status": DocStatus.FAILED,
                            "error_msg": str(e),
                        }
                    }
                )
                self.logger.info(f"Error processing document {file_path}: {str(e)}")
                return False

            # Use provided doc_id or fall back to content-based doc_id
            if doc_id is None:
                doc_id = content_based_doc_id

            # Step 2: Separate text and multimodal content
            text_content, multimodal_items = separate_content(content_list)

            # Step 2.5: Set content source for context extraction in multimodal processing
            if hasattr(self, "set_content_source_for_context") and multimodal_items:
                self.logger.info(
                    "Setting content source for context-aware multimodal processing..."
                )
                self.set_content_source_for_context(
                    content_list, self.config.content_format
                )

            # Step 3: Insert pure text content and multimodal content with all parameters
            if text_content.strip():
                await insert_text_content_with_multimodal_content(
                    self.graphcore,
                    input=text_content,
                    multimodal_content=multimodal_items,
                    file_paths=file_name,
                    split_by_character=split_by_character,
                    split_by_character_only=split_by_character_only,
                    ids=doc_id,
                    scheme_name=scheme_name,
                )

            self.logger.info(f"Document {file_path} processing completed successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error processing document {file_path}: {str(e)}")
            self.logger.debug("Exception details:", exc_info=True)

            # Update doc status to Failed
            await self.graphcore.doc_status.upsert(
                {
                    doc_pre_id: {
                        **current_doc_status,
                        "status": DocStatus.FAILED,
                        "error_msg": str(e),
                    }
                }
            )
            await self.graphcore.doc_status.index_done_callback()

            # Update pipeline status
            if pipeline_status_lock and pipeline_status:
                try:
                    async with pipeline_status_lock:
                        pipeline_status.update({"scan_disabled": False})
                        error_msg = (
                            f"DocThinker processing failed for {file_name}: {str(e)}"
                        )
                        pipeline_status["latest_message"] = error_msg
                        pipeline_status["history_messages"].append(error_msg)
                        pipeline_status["history_messages"].append(
                            "Now is allowed to scan"
                        )
                except Exception as pipeline_update_error:
                    self.logger.error(
                        f"Failed to update pipeline status: {pipeline_update_error}"
                    )

            return False

        finally:
            async with pipeline_status_lock:
                pipeline_status.update({"scan_disabled": False})
                pipeline_status["latest_message"] = (
                    f"DocThinker processing completed for {file_name}"
                )
                pipeline_status["history_messages"].append(
                    f"DocThinker processing completed for {file_name}"
                )
                pipeline_status["history_messages"].append("Now is allowed to scan")

    async def insert_content_list(
        self,
        content_list: List[Dict[str, Any]],
        file_path: str = "unknown_document",
        split_by_character: str | None = None,
        split_by_character_only: bool = False,
        doc_id: str | None = None,
        display_stats: bool = None,
    ):
        """
        Insert content list directly without document parsing

        Args:
            content_list: Pre-parsed content list containing text and multimodal items.
                         Each item should be a dictionary with the following structure:
                         - Text: {"type": "text", "text": "content", "page_idx": 0}
                         - Image: {"type": "image", "img_path": "/absolute/path/to/image.jpg",
                                  "image_caption": ["caption"], "image_footnote": ["note"], "page_idx": 1}
                         - Table: {"type": "table", "table_body": "markdown table",
                                  "table_caption": ["caption"], "table_footnote": ["note"], "page_idx": 2}
                         - Equation: {"type": "equation", "latex": "LaTeX formula",
                                     "text": "description", "page_idx": 3}
                         - Generic: {"type": "custom_type", "content": "any content", "page_idx": 4}
            file_path: Reference file path/name for citation (defaults to "unknown_document")
            split_by_character: Optional character to split the text by
            split_by_character_only: If True, split only by the specified character
            doc_id: Optional document ID, if not provided will be generated from content
            display_stats: Whether to display content statistics (defaults to config.display_content_stats)

        Note:
            - img_path must be an absolute path to the image file
            - page_idx represents the page number where the content appears (0-based indexing)
            - Items are processed in the order they appear in the list
        """
        # Ensure GraphCore is initialized
        await self._ensure_graphcore.coregraph_initialized()

        # Use config defaults if not provided
        if display_stats is None:
            display_stats = self.config.display_content_stats

        self.logger.info(
            f"Starting direct content list insertion for: {file_path} ({len(content_list)} items)"
        )

        # Generate doc_id based on content if not provided
        if doc_id is None:
            doc_id = self._generate_content_based_doc_id(content_list)

        # Display content statistics if requested
        if display_stats:
            self.logger.info("\nContent Information:")
            self.logger.info(f"* Total blocks in content_list: {len(content_list)}")

            # Count elements by type
            block_types: Dict[str, int] = {}
            for block in content_list:
                if isinstance(block, dict):
                    block_type = block.get("type", "unknown")
                    if isinstance(block_type, str):
                        block_types[block_type] = block_types.get(block_type, 0) + 1

            self.logger.info("* Content block types:")
            for block_type, count in block_types.items():
                self.logger.info(f"  - {block_type}: {count}")

        # Step 1: Separate text and multimodal content
        text_content, multimodal_items = separate_content(content_list)

        # Step 1.5: Set content source for context extraction in multimodal processing
        if hasattr(self, "set_content_source_for_context") and multimodal_items:
            self.logger.info(
                "Setting content source for context-aware multimodal processing..."
            )
            self.set_content_source_for_context(
                content_list, self.config.content_format
            )

        # Step 2: Insert pure text content with all parameters
        if text_content.strip():
            file_name = os.path.basename(file_path)
            await insert_text_content(
                self.graphcore,
                input=text_content,
                file_paths=file_name,
                split_by_character=split_by_character,
                split_by_character_only=split_by_character_only,
                ids=doc_id,
            )

        # Step 3: Process multimodal content (using specialized processors)
        if multimodal_items:
            await self._process_multimodal_content(multimodal_items, file_path, doc_id)
        else:
            # If no multimodal content, mark multimodal processing as complete
            # This ensures the document status properly reflects completion of all processing
            await self._mark_multimodal_processing_complete(doc_id)
            self.logger.debug(
                f"No multimodal content found in document {doc_id}, marked multimodal processing as complete"
            )

        self.logger.info(f"Content list insertion complete for: {file_path}")

