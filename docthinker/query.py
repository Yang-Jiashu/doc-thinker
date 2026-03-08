"""
Query functionality for DocThinker

Contains all query-related methods for both text and multimodal queries
"""
#处理文本和多模态查询，并将图像路径传给vlm,并生成检索到的图像内容的增强化描述。
import json
import hashlib
import re
import os
from datetime import datetime
from typing import Dict, List, Any, Set, Tuple, Optional
from pathlib import Path
from graphcore.coregraph import QueryParam
from graphcore.coregraph.utils import always_get_an_event_loop
from docthinker.prompt import PROMPTS
from docthinker.image_assets import ImageAssetTable
from docthinker.utils import (
    get_processor_for_type,
    encode_image_to_base64,
    validate_image_file,
)


class QueryMixin:
    """QueryMixin class containing query functionality for DocThinker"""

    def _generate_multimodal_cache_key(
        self, query: str, multimodal_content: List[Dict[str, Any]], mode: str, **kwargs
    ) -> str:
        """
        Generate cache key for multimodal query

        Args:
            query: Base query text
            multimodal_content: List of multimodal content
            mode: Query mode
            **kwargs: Additional parameters

        Returns:
            str: Cache key hash
        """
        # Create a normalized representation of the query parameters
        cache_data = {
            "query": query.strip(),
            "mode": mode,
        }

        # Normalize multimodal content for stable caching
        normalized_content = []
        if multimodal_content:
            for item in multimodal_content:
                if isinstance(item, dict):
                    normalized_item = {}
                    for key, value in item.items():
                        # For file paths, use basename to make cache more portable
                        if key in [
                            "img_path",
                            "image_path",
                            "file_path",
                        ] and isinstance(value, str):
                            normalized_item[key] = Path(value).name
                        # For large content, create a hash instead of storing directly
                        elif (
                            key in ["table_data", "table_body"]
                            and isinstance(value, str)
                            and len(value) > 200
                        ):
                            normalized_item[f"{key}_hash"] = hashlib.md5(
                                value.encode()
                            ).hexdigest()
                        else:
                            normalized_item[key] = value
                    normalized_content.append(normalized_item)
                else:
                    normalized_content.append(item)

        cache_data["multimodal_content"] = normalized_content

        # Add relevant kwargs to cache data
        relevant_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k
            in [
                "stream",
                "response_type",
                "top_k",
                "max_tokens",
                "temperature",
                # "only_need_context",
                # "only_need_prompt",
            ]
        }
        cache_data.update(relevant_kwargs)

        # Generate hash from the cache data
        cache_str = json.dumps(cache_data, sort_keys=True, ensure_ascii=False)
        cache_hash = hashlib.md5(cache_str.encode()).hexdigest()

        return f"multimodal_query:{cache_hash}"

    async def aquery(self, query: str, mode: str = "mix", **kwargs) -> str:
        """
        Pure text query - directly calls GraphCore's query functionality

        Args:
            query: Query text
            mode: Query mode ("local", "global", "hybrid", "naive", "mix", "bypass")
            **kwargs: Other query parameters, will be passed to QueryParam
                - vlm_enhanced: bool, default True when vision_model_func is available.
                  If True, will parse image paths in retrieved context and replace them
                  with base64 encoded images for VLM processing.
                - graph_traversal_hops: int, default 0. If 1 or 2, expands retrieval by
                  graph BFS (true graph RAG). 0 = vector-only retrieval.

        Returns:
            str: Query result
        """
        if self.graphcore is None:
            raise ValueError(
                "No GraphCore instance available. Please process documents first or provide a pre-initialized GraphCore instance."
            )

        # Reset evidence tracker
        self._last_query_evidence = None

        # Query-time image activation controls (consume custom args before QueryParam)
        enable_image_asset_activation = kwargs.pop(
            "enable_image_asset_activation", True
        )
        try:
            image_activation_threshold = float(
                kwargs.pop(
                    "image_activation_threshold",
                    os.getenv("IMAGE_ACTIVATION_THRESHOLD", "0.62"),
                )
            )
        except Exception:
            image_activation_threshold = 0.62
        try:
            image_activation_top_k = int(
                kwargs.pop(
                    "image_activation_top_k",
                    os.getenv("IMAGE_ACTIVATION_TOP_K", "3"),
                )
            )
        except Exception:
            image_activation_top_k = 3
        image_activation_top_k = max(1, image_activation_top_k)

        # Check if VLM enhanced query should be used
        vlm_enhanced = kwargs.pop("vlm_enhanced", None)

        # Auto-determine VLM enhanced based on availability
        if vlm_enhanced is None:
            vlm_enhanced = (
                hasattr(self, "vision_model_func")
                and self.vision_model_func is not None
            )

        # Use VLM enhanced query if enabled and available
        if (
            vlm_enhanced
            and hasattr(self, "vision_model_func")
            and self.vision_model_func
        ):
            result = await self.aquery_vlm_enhanced(
                query,
                mode=mode,
                enable_image_asset_activation=enable_image_asset_activation,
                image_activation_threshold=image_activation_threshold,
                image_activation_top_k=image_activation_top_k,
                **kwargs,
            )
        elif vlm_enhanced and (
            not hasattr(self, "vision_model_func") or not self.vision_model_func
        ):
            self.logger.warning(
                "VLM enhanced query requested but vision_model_func is not available, falling back to normal query"
            )
            result = await self._execute_text_query(query, mode, **kwargs)
        else:
            result = await self._execute_text_query(query, mode, **kwargs)
        
        # Add query and result to knowledge base
        if hasattr(self, 'add_knowledge_entry'):
            # Add question to knowledge base
            question_entry_id = self.add_knowledge_entry(
                content=query,
                entry_type='question',
                metadata={
                    'query_mode': mode,
                    'vlm_enhanced': vlm_enhanced,
                    'timestamp': str(datetime.now())
                }
            )
            
            # Add answer to knowledge base
            answer_entry_id = self.add_knowledge_entry(
                content=result,
                entry_type='answer',
                metadata={
                    'question_id': question_entry_id,
                    'timestamp': str(datetime.now())
                }
            )
            
            self.logger.info(f"Added query and answer to knowledge base: question_id={question_entry_id}, answer_id={answer_entry_id}")

        self.logger.info("Text query completed")
        return result
    
    async def _execute_text_query(self, query: str, mode: str, **kwargs) -> str:
        """
        Execute a pure text query without knowledge base logging
        """
        # Create query parameters
        query_param = QueryParam(mode=mode, **kwargs)

        self.logger.info(f"Executing text query: {query[:100]}...")
        self.logger.info(f"Query mode: {mode}")

        # Call GraphCore's query method
        result = await self.graphcore.aquery(query, param=query_param)
        self._last_query_evidence = {"raw_prompt": None, "image_paths": []}
        return result

    async def aquery_with_multimodal(
        self,
        query: str,
        multimodal_content: List[Dict[str, Any]] = None,
        mode: str = "mix",
        **kwargs,
    ) -> str:
        """
        Multimodal query - combines text and multimodal content for querying

        Args:
            query: Base query text
            multimodal_content: List of multimodal content, each element contains:
                - type: Content type ("image", "table", "equation", etc.)
                - Other fields depend on type (e.g., img_path, table_data, latex, etc.)
            mode: Query mode ("local", "global", "hybrid", "naive", "mix", "bypass")
            **kwargs: Other query parameters, will be passed to QueryParam

        Returns:
            str: Query result

        Examples:
            # Pure text query
            result = await rag.query_with_multimodal("What is machine learning?")

            # Image query
            result = await rag.query_with_multimodal(
                "Analyze the content in this image",
                multimodal_content=[{
                    "type": "image",
                    "img_path": "./image.jpg"
                }]
            )

            # Table query
            result = await rag.query_with_multimodal(
                "Analyze the data trends in this table",
                multimodal_content=[{
                    "type": "table",
                    "table_data": "Name,Age\nAlice,25\nBob,30"
                }]
            )
        """
        # Ensure CoreGraph is initialized
        await self._ensure_graphcore_initialized()

        self.logger.info(f"Executing multimodal query: {query[:100]}...")
        self.logger.info(f"Query mode: {mode}")

        # If no multimodal content, fallback to pure text query
        if not multimodal_content:
            self.logger.info("No multimodal content provided, executing text query")
            return await self.aquery(query, mode=mode, **kwargs)

        # Generate cache key for multimodal query
        cache_key = self._generate_multimodal_cache_key(
            query, multimodal_content, mode, **kwargs
        )

        # Check cache if available and enabled
        cached_result = None
        if (
            hasattr(self, "graphcore")
            and self.graphcore
            and hasattr(self.graphcore, "llm_response_cache")
            and self.graphcore.llm_response_cache
        ):
            if self.graphcore.llm_response_cache.global_config.get(
                "enable_llm_cache", True
            ):
                try:
                    cached_result = await self.graphcore.llm_response_cache.get_by_id(
                        cache_key
                    )
                    if cached_result and isinstance(cached_result, dict):
                        result_content = cached_result.get("return")
                        if result_content:
                            self.logger.info(
                                f"Multimodal query cache hit: {cache_key[:16]}..."
                            )
                            return result_content
                except Exception as e:
                    self.logger.debug(f"Error accessing multimodal query cache: {e}")

        # Process multimodal content to generate enhanced query text
        enhanced_query = await self._process_multimodal_query_content(
            query, multimodal_content
        )

        self.logger.info(
            f"Generated enhanced query length: {len(enhanced_query)} characters"
        )

        # Execute enhanced query
        result = await self.aquery(enhanced_query, mode=mode, **kwargs)

        # Save to cache if available and enabled
        if (
            hasattr(self, "graphcore")
            and self.graphcore
            and hasattr(self.graphcore, "llm_response_cache")
            and self.graphcore.llm_response_cache
        ):
            if self.graphcore.llm_response_cache.global_config.get(
                "enable_llm_cache", True
            ):
                try:
                    # Create cache entry for multimodal query
                    cache_entry = {
                        "return": result,
                        "cache_type": "multimodal_query",
                        "original_query": query,
                        "multimodal_content_count": len(multimodal_content),
                        "mode": mode,
                    }

                    await self.graphcore.llm_response_cache.upsert(
                        {cache_key: cache_entry}
                    )
                    self.logger.info(
                        f"Saved multimodal query result to cache: {cache_key[:16]}..."
                    )
                except Exception as e:
                    self.logger.debug(f"Error saving multimodal query to cache: {e}")

        # Ensure cache is persisted to disk
        if (
            hasattr(self, "graphcore")
            and self.graphcore
            and hasattr(self.graphcore, "llm_response_cache")
            and self.graphcore.llm_response_cache
        ):
            try:
                await self.graphcore.llm_response_cache.index_done_callback()
            except Exception as e:
                self.logger.debug(f"Error persisting multimodal query cache: {e}")

        self.logger.info("Multimodal query completed")
        return result

    def get_last_query_evidence(self) -> Dict[str, Any] | None:
        """Expose metadata from the most recent query execution."""
        return getattr(self, "_last_query_evidence", None)

    async def aquery_vlm_enhanced(
        self,
        query: str,
        mode: str = "mix",
        *,
        enable_image_asset_activation: bool = True,
        image_activation_threshold: float = 0.62,
        image_activation_top_k: int = 3,
        **kwargs,
    ) -> str:
        """
        VLM enhanced query - replaces image paths in retrieved context with base64 encoded images for VLM processing

        Args:
            query: User query
            mode: Underlying GraphCore query mode
            **kwargs: Other query parameters

        Returns:
            str: VLM query result
        """
        # Ensure VLM is available
        if not hasattr(self, "vision_model_func") or not self.vision_model_func:
            raise ValueError(
                "VLM enhanced query requires vision_model_func. "
                "Please provide a vision model function when initializing DocThinker."
            )

        # Ensure GraphCore is initialized
        await self._ensure_graphcore_initialized()

        self.logger.info(f"Executing VLM enhanced query: {query[:100]}...")

        # Clear previous image cache
        if hasattr(self, "_current_images_base64"):
            delattr(self, "_current_images_base64")

        # 1. Get original retrieval prompt (without generating final answer)
        query_param = QueryParam(mode=mode, only_need_prompt=True, **kwargs)
        raw_prompt = await self.graphcore.aquery(query, param=query_param)

        self.logger.debug("Retrieved raw prompt from GraphCore")

        activated_assets: List[Dict[str, Any]] = []
        if enable_image_asset_activation:
            activated_assets = await self._activate_image_assets_for_query(
                query=query,
                threshold=image_activation_threshold,
                top_k=max(1, image_activation_top_k),
            )
            if activated_assets:
                raw_prompt = (
                    f"{raw_prompt}\n\n"
                    f"[Activated image assets]\n"
                    f"{self._build_activated_image_paths_block(activated_assets)}"
                )
                self.logger.info(
                    "Activated %s image assets for query-time multimodal fusion",
                    len(activated_assets),
                )

        # 2. Extract and process image paths
        enhanced_prompt, images_found, image_paths = await self._process_image_paths_for_vlm(
            raw_prompt
        )

        if not images_found:
            self.logger.info("No valid images found, falling back to normal query")
            # Fallback to normal query
            query_param = QueryParam(mode=mode, **kwargs)
            fallback = await self.graphcore.aquery(query, param=query_param)
            self._last_query_evidence = {
                "raw_prompt": raw_prompt,
                "image_paths": [],
                "activated_image_assets": activated_assets,
            }
            return fallback

        self.logger.info(f"Processed {images_found} images for VLM")

        twi_runner = getattr(self, "twi_runner", None)
        if twi_runner is not None:
            try:
                twi_results = await twi_runner.arun(query, image_paths)
            except Exception as exc:
                self.logger.warning("TWI runner failed, skipping TWI stage: %s", exc)
                twi_results = []
            if twi_results:
                twi_block = "\n\n".join(result.to_prompt_block() for result in twi_results)
                enhanced_prompt = (
                    f"{enhanced_prompt}\n\n[Think-With-Image 摘要]\n{twi_block}"
                )

        # 3. Build VLM message format
        messages = self._build_vlm_messages_with_images(enhanced_prompt, query)

        # 4. Call VLM for question answering
        result = await self._call_vlm_with_multimodal_content(messages)
        self._last_query_evidence = {
            "raw_prompt": raw_prompt,
            "image_paths": image_paths,
            "activated_image_assets": activated_assets,
        }

        self.logger.info("VLM enhanced query completed")
        return result

    async def _activate_image_assets_for_query(
        self,
        *,
        query: str,
        threshold: float,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        if not query or not hasattr(self, "embedding_func") or self.embedding_func is None:
            return []

        table = ImageAssetTable(self.working_dir)
        if not table.load():
            return []

        try:
            query_vec = await self.embedding_func([query])
            selected = table.select_activated_by_embedding(
                query_embedding=query_vec,
                threshold=float(threshold),
                top_k=max(1, int(top_k)),
            )
        except Exception as exc:
            self.logger.warning(f"Image asset activation failed: {exc}")
            return []

        activated: List[Dict[str, Any]] = []
        for item in selected:
            path = str(item.get("stored_path") or "").strip()
            if not path:
                continue
            if not Path(path).exists():
                continue
            activated.append(item)
        return activated

    @staticmethod
    def _build_activated_image_paths_block(records: List[Dict[str, Any]]) -> str:
        if not records:
            return ""
        lines = ["Image Paths:"]
        for rec in records:
            path = str(rec.get("stored_path") or "").strip()
            if path:
                lines.append(path)
        return "\n".join(lines)

    async def _process_multimodal_query_content(
        self, base_query: str, multimodal_content: List[Dict[str, Any]]
    ) -> str:
        """
        Process multimodal query content to generate enhanced query text

        Args:
            base_query: Base query text
            multimodal_content: List of multimodal content

        Returns:
            str: Enhanced query text
        """
        self.logger.info("Starting multimodal query content processing...")

        enhanced_parts = [f"User query: {base_query}"]

        for i, content in enumerate(multimodal_content):
            content_type = content.get("type", "unknown")
            self.logger.info(
                f"Processing {i+1}/{len(multimodal_content)} multimodal content: {content_type}"
            )

            try:
                # Get appropriate processor
                processor = get_processor_for_type(self.modal_processors, content_type)

                if processor:
                    # Generate content description
                    description = await self._generate_query_content_description(
                        processor, content, content_type
                    )
                    enhanced_parts.append(
                        f"\nRelated {content_type} content: {description}"
                    )
                else:
                    # If no appropriate processor, use basic description
                    basic_desc = str(content)[:200]
                    enhanced_parts.append(
                        f"\nRelated {content_type} content: {basic_desc}"
                    )

            except Exception as e:
                self.logger.error(f"Error processing multimodal content: {str(e)}")
                # Continue processing other content
                continue

        enhanced_query = "\n".join(enhanced_parts)
        enhanced_query += PROMPTS["QUERY_ENHANCEMENT_SUFFIX"]

        self.logger.info("Multimodal query content processing completed")
        return enhanced_query

    async def _generate_query_content_description(
        self, processor, content: Dict[str, Any], content_type: str
    ) -> str:
        """
        Generate content description for query

        Args:
            processor: Multimodal processor
            content: Content data
            content_type: Content type

        Returns:
            str: Content description
        """
        try:
            if content_type == "image":
                return await self._describe_image_for_query(processor, content)
            elif content_type == "table":
                return await self._describe_table_for_query(processor, content)
            elif content_type == "equation":
                return await self._describe_equation_for_query(processor, content)
            else:
                return await self._describe_generic_for_query(
                    processor, content, content_type
                )

        except Exception as e:
            self.logger.error(f"Error generating {content_type} description: {str(e)}")
            return f"{content_type} content: {str(content)[:100]}"

    async def _describe_image_for_query(
        self, processor, content: Dict[str, Any]
    ) -> str:
        """Generate image description for query"""
        image_path = content.get("img_path")
        captions = content.get("image_caption", content.get("img_caption", []))
        footnotes = content.get("image_footnote", content.get("img_footnote", []))

        if image_path and Path(image_path).exists():
            # If image exists, use vision model to generate description
            image_base64 = processor._encode_image_to_base64(image_path)
            if image_base64:
                import mimetypes

                mime_type, _ = mimetypes.guess_type(image_path)
                if not mime_type:
                    mime_type = "image/png"
                image_data_uri = f"data:{mime_type};base64,{image_base64}"

                prompt = PROMPTS["QUERY_IMAGE_DESCRIPTION"]
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": image_data_uri}},
                        ],
                    }
                ]
                description = await processor.modal_caption_func(
                    prompt,
                    messages=messages,
                    system_prompt=PROMPTS["QUERY_IMAGE_ANALYST_SYSTEM"],
                )
                return description

        # If image doesn't exist or processing failed, use existing information
        parts = []
        if image_path:
            parts.append(f"Image path: {image_path}")
        if captions:
            parts.append(f"Image captions: {', '.join(captions)}")
        if footnotes:
            parts.append(f"Image footnotes: {', '.join(footnotes)}")

        return "; ".join(parts) if parts else "Image content information incomplete"

    async def _describe_table_for_query(
        self, processor, content: Dict[str, Any]
    ) -> str:
        """Generate table description for query"""
        table_data = content.get("table_data", "")
        table_caption = content.get("table_caption", "")

        prompt = PROMPTS["QUERY_TABLE_ANALYSIS"].format(
            table_data=table_data, table_caption=table_caption
        )

        description = await processor.modal_caption_func(
            prompt, system_prompt=PROMPTS["QUERY_TABLE_ANALYST_SYSTEM"]
        )

        return description

    async def _describe_equation_for_query(
        self, processor, content: Dict[str, Any]
    ) -> str:
        """Generate equation description for query"""
        latex = content.get("latex", "")
        equation_caption = content.get("equation_caption", "")

        prompt = PROMPTS["QUERY_EQUATION_ANALYSIS"].format(
            latex=latex, equation_caption=equation_caption
        )

        description = await processor.modal_caption_func(
            prompt, system_prompt=PROMPTS["QUERY_EQUATION_ANALYST_SYSTEM"]
        )

        return description

    async def _describe_generic_for_query(
        self, processor, content: Dict[str, Any], content_type: str
    ) -> str:
        """Generate generic content description for query"""
        content_str = str(content)

        prompt = PROMPTS["QUERY_GENERIC_ANALYSIS"].format(
            content_type=content_type, content_str=content_str
        )

        description = await processor.modal_caption_func(
            prompt,
            system_prompt=PROMPTS["QUERY_GENERIC_ANALYST_SYSTEM"].format(
                content_type=content_type
            ),
        )

        return description

    async def _process_image_paths_for_vlm(
        self, prompt: str | None
    ) -> tuple[str, int, List[str]]:
        """
        Process image paths in prompt, keeping original paths and adding VLM markers

        Args:
            prompt: Original prompt

        Returns:
            tuple: (processed prompt, image count)
        """
        if not isinstance(prompt, str):
            self.logger.warning(
                "VLM prompt is empty or None, skipping image path normalization"
            )
            return "", 0, []

        multi_paths_pattern = re.compile(
            r"Image Paths:\s*((?:[^\r\n]*?\.(?:jpg|jpeg|png|gif|bmp|webp|tiff|tif)\s*(?:\r?\n|$))+)",
            re.IGNORECASE,
        )

        def _normalize_paths_block(match: re.Match) -> str:
            block = match.group(1)
            candidates = re.findall(
                r"([^\r\n]*?\.(?:jpg|jpeg|png|gif|bmp|webp|tiff|tif))",
                block,
                flags=re.IGNORECASE,
            )
            normalized = [
                f"Image Path: {candidate.strip()}"
                for candidate in candidates
                if candidate.strip()
            ]
            if not normalized:
                return match.group(0)
            suffix = "\n" if block.endswith(("\r\n", "\n")) else ""
            return "\n".join(normalized) + suffix

        enhanced_prompt = multi_paths_pattern.sub(_normalize_paths_block, prompt)
        images_processed = 0
        processed_image_paths: List[str] = []

        # Initialize image cache
        self._current_images_base64 = []

        # Enhanced regex pattern for matching image paths
        # Matches only the path ending with image file extensions
        image_path_pattern = (
            r"Image Path[s]?:\s*([^\r\n]*?\.(?:jpg|jpeg|png|gif|bmp|webp|tiff|tif))"
        )

        # First, let's see what matches we find
        matches = re.findall(image_path_pattern, enhanced_prompt)
        self.logger.info(f"Found {len(matches)} image path matches in prompt")

        def replace_image_path(match):
            nonlocal images_processed
            image_path = match.group(1).strip().strip('"').strip("'").strip("`")
            self.logger.debug(f"Processing image path: '{image_path}'")

            # Validate path format (basic check)
            if not image_path or len(image_path) < 3:
                self.logger.warning(f"Invalid image path format: {image_path}")
                return match.group(0)  # Keep original

            # Use utility function to validate image file
            self.logger.debug(f"Calling validate_image_file for: {image_path}")
            is_valid = validate_image_file(image_path)
            self.logger.debug(f"Validation result for {image_path}: {is_valid}")

            if not is_valid:
                # Attempt relocation by filename within candidate parent and MINERU_ROOT
                candidate = Path(image_path)
                fname = candidate.name
                relocated = None
                search_roots = []
                if candidate.parent:
                    search_roots.append(candidate.parent)
                base_root = os.getenv("MINERU_ROOT")
                if base_root and Path(base_root).exists():
                    search_roots.append(Path(base_root))
                for root in search_roots:
                    try:
                        for r, _dirs, files in os.walk(root):
                            if fname in files:
                                relocated = Path(r) / fname
                                break
                        if relocated:
                            break
                    except Exception:
                        pass
                if relocated and validate_image_file(str(relocated)):
                    image_path = relocated.as_posix()
                    self.logger.info(f"Relocated image path to: {image_path}")
                else:
                    self.logger.warning(f"Image validation failed for: {image_path}")
                    return match.group(0)  # Keep original if validation fails

            try:
                # Encode image to base64 using utility function
                self.logger.debug(f"Attempting to encode image: {image_path}")
                image_base64 = encode_image_to_base64(image_path)
                if image_base64:
                    images_processed += 1
                    # Save base64 to instance variable for later use
                    self._current_images_base64.append(image_base64)
                    processed_image_paths.append(image_path)

                    # Keep original path info and add VLM marker
                    result = f"Image Path: {image_path}\n[VLM_IMAGE_{images_processed}]"
                    self.logger.debug(
                        f"Successfully processed image {images_processed}: {image_path}"
                    )
                    return result
                else:
                    self.logger.error(f"Failed to encode image: {image_path}")
                    return match.group(0)  # Keep original if encoding failed

            except Exception as e:
                self.logger.error(f"Failed to process image {image_path}: {e}")
                return match.group(0)  # Keep original

        # Execute replacement
        enhanced_prompt = re.sub(
            image_path_pattern, replace_image_path, enhanced_prompt
        )

        return enhanced_prompt, images_processed, processed_image_paths

    async def aquery_multi_document_enhanced(self, query: str, mode: str = "mix", **kwargs) -> Dict[str, Any]:
        """
        Multi-document enhanced query - uses knowledge graph to find related documents and enhance query context

        Args:
            query: User query
            mode: Underlying GraphCore query mode
            **kwargs: Other query parameters

        Returns:
            Dict[str, Any]: Query result with related documents information
        """
        self.logger.info(f"Executing multi-document enhanced query: {query[:100]}...")
        
        use_fast_kg = kwargs.pop("use_fast_kg", True)
        fast_kg_only = kwargs.pop("fast_kg_only", False)
        fast_kg_hops = kwargs.pop("fast_kg_hops", 1)
        fast_kg_max_docs = kwargs.pop("fast_kg_max_docs", 50)
        fast_kg_max_entities = kwargs.pop("fast_kg_max_entities", 12)
        fast_kg_min_score = kwargs.pop("fast_kg_min_score", 0.2)

        entities: List[str] = []
        related_docs: Set[str] = set()

        if use_fast_kg and hasattr(self, "knowledge_graph") and self.knowledge_graph is not None:
            fast_related_docs, fast_entities = self._find_related_documents_fast(
                query,
                hops=fast_kg_hops,
                max_docs=fast_kg_max_docs,
                max_entities=fast_kg_max_entities,
                min_score=fast_kg_min_score,
            )
            self.logger.info(f"Fast KG entities: {fast_entities}")
            self.logger.info(f"Fast KG related documents: {fast_related_docs}")
            entities.extend(fast_entities)
            related_docs.update(fast_related_docs)

        if not fast_kg_only:
            llm_entities = await self._extract_entities_from_query(query)
            self.logger.info(f"Extracted entities from query: {llm_entities}")
            llm_related_docs = self._find_related_documents(llm_entities)
            self.logger.info(f"Found related documents from LLM entities: {llm_related_docs}")
            
            # Merge results
            for e in llm_entities:
                if e not in entities:
                    entities.append(e)
            related_docs.update(llm_related_docs)
        
        # Step 3: Build enhanced query with related document context
        enhanced_query = await self._build_enhanced_query(query, entities, related_docs)
        
        # Step 4: Execute query with enhanced context
        result = await self.aquery(enhanced_query, mode=mode, **kwargs)
        
        # Step 5: Return result with related documents information
        return {
            "answer": result,
            "related_documents": list(related_docs),
            "extracted_entities": entities
        }
    
    async def _extract_entities_from_query(self, query: str) -> List[str]:
        """
        Extract entities from query using LLM
        
        Args:
            query: User query
            
        Returns:
            List[str]: Extracted entities
        """
        prompt = f"""Extract key entities from this question. Return only the entity names as a JSON array.

Question: {query}

Example output: ["entity1", "entity2", "entity3"]
"""
        
        try:
            response = await self.llm_model_func(prompt, system_prompt="You are an expert entity extractor.")
            entities = json.loads(response)
            return entities if isinstance(entities, list) else []
        except Exception as e:
            self.logger.error(f"Error extracting entities: {e}")
            return []
    
    def _find_related_documents(self, entities: List[str]) -> Set[str]:
        """
        Find related documents from knowledge graph based on entities
        
        Args:
            entities: List of entities
            
        Returns:
            Set[str]: Related document IDs
        """
        related_docs = set()
        
        if not hasattr(self, 'knowledge_graph') or self.knowledge_graph is None:
            return related_docs
        
        for entity_name in entities:
            entity = self.knowledge_graph.get_entity_by_name(entity_name)
            if entity:
                # Get all relationships involving this entity
                relationships = self.knowledge_graph.get_relationships_by_entity(entity.id)
                for relationship in relationships:
                    # Add all documents that mention these relationships
                    related_docs.update(relationship.document_ids)
        
        return related_docs

    def _find_related_documents_fast(
        self,
        query: str,
        hops: int = 1,
        max_docs: int = 50,
        max_entities: int = 12,
        min_score: float = 0.2,
    ) -> Tuple[Set[str], List[str]]:
        if not hasattr(self, "knowledge_graph") or self.knowledge_graph is None:
            return set(), []
        if not hasattr(self.knowledge_graph, "fast_related_documents"):
            return set(), []
        related_docs, entities = self.knowledge_graph.fast_related_documents(
            query=query,
            hops=hops,
            max_docs=max_docs,
            max_entities=max_entities,
            min_score=min_score,
        )
        return related_docs, entities
    
    async def _build_enhanced_query(self, original_query: str, entities: List[str], related_docs: Set[str]) -> str:
        """
        Build enhanced query with related document context
        
        Args:
            original_query: User's original query
            entities: Extracted entities
            related_docs: Related document IDs
            
        Returns:
            str: Enhanced query
        """
        if not related_docs:
            return original_query
        
        # Get context from related documents
        related_context = await self._get_context_from_related_docs(related_docs, entities)
        
        enhanced_query = f"""You are answering a question. Please consider the following context from related documents when formulating your answer.

Related Documents Context:
{related_context}

Original Question: {original_query}

Please provide a comprehensive answer that integrates information from all relevant documents."""
        
        return enhanced_query
    
    async def _get_context_from_related_docs(self, related_docs: Set[str], entities: List[str]) -> str:
        """
        Get relevant context from related documents
        
        Args:
            related_docs: Related document IDs
            entities: Extracted entities
            
        Returns:
            str: Combined context from related documents
        """
        contexts = []
        
        for doc_id in related_docs:
            # For each related document, get relevant context using entities
            prompt = f"""Extract paragraphs from document {doc_id} that mention any of these entities: {entities}.
            Return only the relevant paragraphs, not the entire document."""
            
            # In a real implementation, you would retrieve actual content from the document
            # For now, we'll use a simplified approach
            context = f"[Context from document {doc_id}: This document contains information about {', '.join(entities)}]"
            contexts.append(context)
        
        return "\n\n".join(contexts)
    
    def _build_vlm_messages_with_images(
        self, enhanced_prompt: str, user_query: str
    ) -> List[Dict]:
        """
        Build VLM message format, using markers to correspond images with text positions

        Args:
            enhanced_prompt: Enhanced prompt with image markers
            user_query: User query

        Returns:
            List[Dict]: VLM message format
        """
        images_base64 = getattr(self, "_current_images_base64", [])

        if not images_base64:
            # Pure text mode
            return [
                {
                    "role": "user",
                    "content": f"Context:\n{enhanced_prompt}\n\nUser Question: {user_query}",
                }
            ]

        # Build multimodal content
        content_parts = []

        # Split text at image markers and insert images
        text_parts = enhanced_prompt.split("[VLM_IMAGE_")

        for i, text_part in enumerate(text_parts):
            if i == 0:
                # First text part
                if text_part.strip():
                    content_parts.append({"type": "text", "text": text_part})
            else:
                # Find marker number and insert corresponding image
                marker_match = re.match(r"(\d+)\](.*)", text_part, re.DOTALL)
                if marker_match:
                    image_num = (
                        int(marker_match.group(1)) - 1
                    )  # Convert to 0-based index
                    remaining_text = marker_match.group(2)

                    # Insert corresponding image
                    if 0 <= image_num < len(images_base64):
                        content_parts.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{images_base64[image_num]}"
                                },
                            }
                        )

                    # Insert remaining text
                    if remaining_text.strip():
                        content_parts.append({"type": "text", "text": remaining_text})

        # Add user question
        content_parts.append(
            {
                "type": "text",
                "text": f"\n\nUser Question: {user_query}\n\nPlease answer based on the context and images provided.",
            }
        )

        return [
            {
                "role": "system",
                "content": "You are a helpful assistant that can analyze both text and image content to provide comprehensive answers.",
            },
            {"role": "user", "content": content_parts},
        ]

    async def _call_vlm_with_multimodal_content(self, messages: List[Dict]) -> str:
        """
        Call VLM to process multimodal content

        Args:
            messages: VLM message format

        Returns:
            str: VLM response result
        """
        try:
            user_message = messages[1]
            content = user_message["content"]
            system_prompt = messages[0]["content"]

            if isinstance(content, str):
                result = await self.vision_model_func(
                    content, system_prompt=system_prompt, max_tokens=8192,
                )
            else:
                result = await self.vision_model_func(
                    "",
                    messages=messages,
                    max_tokens=8192,
                )

            return result

        except Exception as e:
            self.logger.error(f"VLM call failed: {e}")
            raise

    # Synchronous versions of query methods
    def query(self, query: str, mode: str = "mix", **kwargs) -> str:
        """
        Synchronous version of pure text query

        Args:
            query: Query text
            mode: Query mode ("local", "global", "hybrid", "naive", "mix", "bypass")
            **kwargs: Other query parameters, will be passed to QueryParam
                - vlm_enhanced: bool, default True when vision_model_func is available.
                  If True, will parse image paths in retrieved context and replace them
                  with base64 encoded images for VLM processing.

        Returns:
            str: Query result
        """
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.aquery(query, mode=mode, **kwargs))

    def query_with_multimodal(
        self,
        query: str,
        multimodal_content: List[Dict[str, Any]] = None,
        mode: str = "mix",
        **kwargs,
    ) -> str:
        """
        Synchronous version of multimodal query

        Args:
            query: Base query text
            multimodal_content: List of multimodal content, each element contains:
                - type: Content type ("image", "table", "equation", etc.)
                - Other fields depend on type (e.g., img_path, table_data, latex, etc.)
            mode: Query mode ("local", "global", "hybrid", "naive", "mix", "bypass")
            **kwargs: Other query parameters, will be passed to QueryParam

        Returns:
            str: Query result
        """
        loop = always_get_an_event_loop()
        return loop.run_until_complete(
            self.aquery_with_multimodal(query, multimodal_content, mode=mode, **kwargs)
        )

    def query_multi_document_enhanced(
        self,
        query: str,
        mode: str = "mix",
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Synchronous version of multi-document enhanced query

        Args:
            query: User query
            mode: Underlying GraphCore query mode
            **kwargs: Other query parameters

        Returns:
            Dict[str, Any]: Query result with related documents information
        """
        loop = always_get_an_event_loop()
        return loop.run_until_complete(
            self.aquery_multi_document_enhanced(query, mode=mode, **kwargs)
        )
    
    async def aquery_with_knowledge_reasoning(
        self,
        query: str,
        mode: str = "mix",
        knowledge_base_name: str = "global",
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Query with knowledge base reasoning capabilities

        Args:
            query: User query
            mode: Underlying GraphCore query mode
            knowledge_base_name: Name of knowledge base to use for reasoning
            **kwargs: Other query parameters

        Returns:
            Dict[str, Any]: Query result with reasoning information
        """
        self.logger.info(f"Executing query with knowledge reasoning: {query[:100]}...")
        
        # Step 1: Use GraphCore to get initial answer
        initial_answer = await self.aquery(query, mode=mode, **kwargs)
        
        # Step 2: Use knowledge base to enhance answer with reasoning
        knowledge_enhanced = False
        reasoning_info = {}
        
        if hasattr(self, 'query_knowledge_base_with_reasoning'):
            # Query knowledge base with reasoning
            kb_results, reasoning_info = self.query_knowledge_base_with_reasoning(
                kb_name=knowledge_base_name,
                query_text=query
            )
            
            if kb_results:
                knowledge_enhanced = True
                
                # Enhance answer with knowledge base results
                enhanced_answer = await self._enhance_answer_with_knowledge(
                    query, initial_answer, kb_results, reasoning_info
                )
                
                # Add query and enhanced answer to knowledge base
                if hasattr(self, 'add_knowledge_entry'):
                    question_entry_id = self.add_knowledge_entry(
                        content=query,
                        entry_type='question',
                        metadata={'query_mode': mode, 'timestamp': datetime.now().isoformat()},
                        kb_name=knowledge_base_name
                    )
                    
                    answer_entry_id = self.add_knowledge_entry(
                        content=enhanced_answer,
                        entry_type='answer',
                        metadata={'question_id': question_entry_id, 'is_enhanced': True, 'timestamp': datetime.now().isoformat()},
                        kb_name=knowledge_base_name
                    )
                
                result = {
                    "answer": enhanced_answer,
                    "initial_answer": initial_answer,
                    "knowledge_enhanced": knowledge_enhanced,
                    "reasoning_info": reasoning_info,
                    "kb_results_count": len(kb_results),
                    "kb_name": knowledge_base_name
                }
            else:
                # Use initial answer if no knowledge base results
                result = {
                    "answer": initial_answer,
                    "initial_answer": initial_answer,
                    "knowledge_enhanced": False,
                    "reasoning_info": reasoning_info,
                    "kb_results_count": 0,
                    "kb_name": knowledge_base_name
                }
        else:
            # No knowledge base functionality available
            result = {
                "answer": initial_answer,
                "initial_answer": initial_answer,
                "knowledge_enhanced": False,
                "reasoning_info": {},
                "kb_results_count": 0,
                "kb_name": knowledge_base_name
            }
        
        self.logger.info("Query with knowledge reasoning completed")
        return result
    
    async def _enhance_answer_with_knowledge(
        self,
        query: str,
        initial_answer: str,
        kb_results: List[Dict[str, Any]],
        reasoning_info: Dict[str, Any]
    ) -> str:
        """
        Enhance initial answer with knowledge base results

        Args:
            query: Original query
            initial_answer: Initial answer from GraphCore
            kb_results: Knowledge base results
            reasoning_info: Reasoning information

        Returns:
            str: Enhanced answer
        """
        # Extract relevant content from knowledge base results
        relevant_content = []
        for result in kb_results:
            entry = result.get('entry')
            if entry:
                relevant_content.append(entry.content)
        
        if not relevant_content:
            return initial_answer
        
        # Build enhancement prompt
        knowledge_text = "\n\n".join(relevant_content[:5])
        prompt = f"""
        You are a helpful assistant that enhances answers using additional knowledge.
        Please use the following information to enhance the initial answer to the user's question.
        
        User Question: {query}
        
        Initial Answer: {initial_answer}
        
        Additional Knowledge:
        {knowledge_text}
        
        Please provide an enhanced answer that integrates the initial answer with the additional knowledge. Make sure the answer is comprehensive and accurate.
        """
        
        try:
            enhanced_answer = await self.llm_model_func(prompt, system_prompt="You are an expert answer enhancer.")
            return enhanced_answer
        except Exception as e:
            self.logger.error(f"Error enhancing answer with knowledge: {e}")
            return initial_answer
    
    def query_with_knowledge_reasoning(
        self,
        query: str,
        mode: str = "mix",
        knowledge_base_name: str = "global",
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Synchronous version of query with knowledge reasoning

        Args:
            query: User query
            mode: Underlying GraphCore query mode
            knowledge_base_name: Name of knowledge base to use for reasoning
            **kwargs: Other query parameters

        Returns:
            Dict[str, Any]: Query result with reasoning information
        """
        loop = always_get_an_event_loop()
        return loop.run_until_complete(
            self.aquery_with_knowledge_reasoning(query, mode=mode, knowledge_base_name=knowledge_base_name, **kwargs)
        )
    
    async def aquery_multi_dimension(
        self,
        query: str,
        knowledge_base_name: str = "global",
        entry_types: Optional[List[str]] = None,
        entities: Optional[List[str]] = None,
        metadata_filters: Optional[Dict[str, Any]] = None,
        min_confidence: float = 0.5,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Multi-dimension query using knowledge base

        Args:
            query: User query
            knowledge_base_name: Name of knowledge base to query
            entry_types: List of entry types to filter (e.g., ['entity', 'answer'])
            entities: List of entity IDs to filter
            metadata_filters: Metadata filters (key-value pairs)
            min_confidence: Minimum confidence score for results
            **kwargs: Other query parameters

        Returns:
            Dict[str, Any]: Query result with multi-dimension information
        """
        self.logger.info(f"Executing multi-dimension query: {query[:100]}...")
        
        if not hasattr(self, 'multi_dimension_query'):
            return {"answer": "Multi-dimension query functionality not available", "success": False}
        
        # Perform multi-dimension query on knowledge base
        kb_results = self.multi_dimension_query(
            kb_name=knowledge_base_name,
            query_text=query,
            entry_types=entry_types,
            entities=entities,
            metadata_filters=metadata_filters,
            min_confidence=min_confidence
        )
        
        if kb_results:
            # Generate answer from knowledge base results
            answer = await self._generate_answer_from_kb_results(query, kb_results)
            
            result = {
                "answer": answer,
                "success": True,
                "results_count": len(kb_results),
                "kb_name": knowledge_base_name,
                "filters": {
                    "entry_types": entry_types,
                    "entities": entities,
                    "metadata_filters": metadata_filters,
                    "min_confidence": min_confidence
                }
            }
        else:
            result = {
                "answer": "No results found for your query",
                "success": False,
                "results_count": 0,
                "kb_name": knowledge_base_name,
                "filters": {
                    "entry_types": entry_types,
                    "entities": entities,
                    "metadata_filters": metadata_filters,
                    "min_confidence": min_confidence
                }
            }
        
        self.logger.info("Multi-dimension query completed")
        return result
    
    async def _generate_answer_from_kb_results(
        self,
        query: str,
        kb_results: List[Dict[str, Any]]
    ) -> str:
        """
        Generate answer from knowledge base results

        Args:
            query: User query
            kb_results: Knowledge base results

        Returns:
            str: Generated answer
        """
        # Extract content from results
        result_content = []
        for result in kb_results[:10]:  # Limit to first 10 results
            entry = result.get('entry')
            if entry:
                result_content.append(entry.content)
        
        if not result_content:
            return "No relevant information found in the knowledge base."
        
        # Build prompt for answer generation
        kb_info_text = "\n\n".join(result_content)
        prompt = f"""
        Please answer the user's question using the following information from the knowledge base.
        Make sure your answer is accurate, comprehensive, and directly addresses the question.
        
        User Question: {query}
        
        Knowledge Base Information:
        {kb_info_text}
        
        Answer:
        """
        
        try:
            answer = await self.llm_model_func(prompt, system_prompt="You are an expert knowledge base answer generator.")
            return answer
        except Exception as e:
            self.logger.error(f"Error generating answer from KB results: {e}")
            return "An error occurred while generating the answer from the knowledge base."
    
    def query_multi_dimension(
        self,
        query: str,
        knowledge_base_name: str = "global",
        entry_types: Optional[List[str]] = None,
        entities: Optional[List[str]] = None,
        metadata_filters: Optional[Dict[str, Any]] = None,
        min_confidence: float = 0.5,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Synchronous version of multi-dimension query

        Args:
            query: User query
            knowledge_base_name: Name of knowledge base to query
            entry_types: List of entry types to filter (e.g., ['entity', 'answer'])
            entities: List of entity IDs to filter
            metadata_filters: Metadata filters (key-value pairs)
            min_confidence: Minimum confidence score for results
            **kwargs: Other query parameters

        Returns:
            Dict[str, Any]: Query result with multi-dimension information
        """
        loop = always_get_an_event_loop()
        return loop.run_until_complete(
            self.aquery_multi_dimension(
                query, 
                knowledge_base_name=knowledge_base_name,
                entry_types=entry_types,
                entities=entities,
                metadata_filters=metadata_filters,
                min_confidence=min_confidence,
                **kwargs
            )
        )
