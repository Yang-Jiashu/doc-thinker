#hypergraphrag的主实现流程，调用子文件实现chunking,关系实体抽取，多模态chunk缓存，查询
import asyncio
import json
import os
from tqdm.asyncio import tqdm as tqdm_async
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from functools import partial
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type, Union, cast

from .schemas import StructuredChunk
from ..twi_adapter import ThinkWithImageRunner

from .llm import openai_embedding
from .bltcy_adapter import bltcy_gpt4o_mini_complete
from .operate import (
    chunking_by_token_size,
    extract_entities,
    # local_query,global_query,hybrid_query,
    kg_query
)
from .prompt import PROMPTS
from .utils import (
    EmbeddingFunc,
    compute_mdhash_id,
    limit_async_func_call,
    convert_response_to_json,
    logger,
    set_logger,
)
from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    StorageNameSpace,
    QueryParam,
)

from .storage import (
    JsonKVStorage,
    NanoVectorDBStorage,
    NetworkXStorage,
)

# future KG integrations

# from .kg.ArangoDB_impl import (
#     GraphStorage as ArangoDBStorage
# )


def lazy_external_import(module_name: str, class_name: str):
    """Lazily import a class from an external module based on the package of the caller."""

    # Get the caller's module and package
    import inspect

    caller_frame = inspect.currentframe().f_back
    module = inspect.getmodule(caller_frame)
    package = module.__package__ if module else None

    def import_class(*args, **kwargs):
        import importlib

        # Import the module using importlib
        module = importlib.import_module(module_name, package=package)

        # Get the class from the module and instantiate it
        cls = getattr(module, class_name)
        return cls(*args, **kwargs)

    return import_class


Neo4JStorage = lazy_external_import(".kg.neo4j_impl", "Neo4JStorage")
OracleKVStorage = lazy_external_import(".kg.oracle_impl", "OracleKVStorage")
OracleGraphStorage = lazy_external_import(".kg.oracle_impl", "OracleGraphStorage")
OracleVectorDBStorage = lazy_external_import(".kg.oracle_impl", "OracleVectorDBStorage")
MilvusVectorDBStorge = lazy_external_import(".kg.milvus_impl", "MilvusVectorDBStorge")
MongoKVStorage = lazy_external_import(".kg.mongo_impl", "MongoKVStorage")
ChromaVectorDBStorage = lazy_external_import(".kg.chroma_impl", "ChromaVectorDBStorage")
TiDBKVStorage = lazy_external_import(".kg.tidb_impl", "TiDBKVStorage")
TiDBVectorDBStorage = lazy_external_import(".kg.tidb_impl", "TiDBVectorDBStorage")


def always_get_an_event_loop() -> asyncio.AbstractEventLoop:
    """
    Ensure that there is always an event loop available.

    This function tries to get the current event loop. If the current event loop is closed or does not exist,
    it creates a new event loop and sets it as the current event loop.

    Returns:
        asyncio.AbstractEventLoop: The current or newly created event loop.
    """
    try:
        # Try to get the current event loop
        current_loop = asyncio.get_event_loop()
        if current_loop.is_closed():
            raise RuntimeError("Event loop is closed.")
        return current_loop

    except RuntimeError:
        # If no event loop exists or it is closed, create a new one
        logger.info("Creating a new event loop in main thread.")
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        return new_loop


@dataclass
class HyperGraphRAG:
    working_dir: str = field(
        default_factory=lambda: f"hypergraphrag_cache_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
    )
    # Default not to use embedding cache
    embedding_cache_config: dict = field(
        default_factory=lambda: {
            "enabled": False,
            "similarity_threshold": 0.95,
            "use_llm_check": False,
        }
    )
    kv_storage: str = field(default="JsonKVStorage")
    vector_storage: str = field(default="NanoVectorDBStorage")
    graph_storage: str = field(default="NetworkXStorage")

    current_log_level = logger.level
    log_level: str = field(default=current_log_level)

    # text chunking
    chunk_token_size: int = 1200
    chunk_overlap_token_size: int = 100
    # Use a tiktoken-supported model name for tokenizer mapping
    tiktoken_model_name: str = "gpt-4o-mini"

    # entity extraction
    entity_extract_max_gleaning: int = 2
    entity_summary_to_max_tokens: int = 500

    # node embedding
    node_embedding_algorithm: str = "node2vec"
    node2vec_params: dict = field(
        default_factory=lambda: {
            "dimensions": 1536,
            "num_walks": 10,
            "walk_length": 40,
            "window_size": 2,
            "iterations": 3,
            "random_seed": 3,
        }
    )

    # embedding_func: EmbeddingFunc = field(default_factory=lambda:hf_embedding)
    embedding_func: EmbeddingFunc = field(default_factory=lambda: openai_embedding)
    embedding_batch_num: int = 32
    embedding_func_max_async: int = 12

    # LLM
    llm_model_func: callable = bltcy_gpt4o_mini_complete  # hf_model_complete#
    llm_model_name: str = "qwen-plus"
    llm_model_max_token_size: int = 32768
    llm_model_max_async: int = 12
    llm_model_kwargs: dict = field(default_factory=dict)
    
    # graph construction mode
    graph_construction_mode: str = "llm" # or "linear"
    spacy_model: str = "en_core_web_sm"

    # storage
    vector_db_storage_cls_kwargs: dict = field(default_factory=dict)

    enable_llm_cache: bool = True

    # extension
    addon_params: dict = field(default_factory=dict)
    convert_response_to_json_func: callable = convert_response_to_json
    vlm_client: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        log_file = os.path.join("hypergraphrag.log")
        set_logger(log_file)
        logger.setLevel(self.log_level)

        logger.info(f"Logger initialized for working directory: {self.working_dir}")

        _print_config = ",\n  ".join([f"{k} = {v}" for k, v in asdict(self).items()])
        logger.debug(f"HyperGraphRAG init with param:\n  {_print_config}\n")

        self.key_string_value_json_storage_cls: Type[BaseKVStorage] = (
            self._get_storage_class()[self.kv_storage]
        )
        self.vector_db_storage_cls: Type[BaseVectorStorage] = self._get_storage_class()[
            self.vector_storage
        ]
        self.graph_storage_cls: Type[BaseGraphStorage] = self._get_storage_class()[
            self.graph_storage
        ]

        if not os.path.exists(self.working_dir):
            logger.info(f"Creating working directory {self.working_dir}")
            os.makedirs(self.working_dir)

        self.llm_response_cache = (
            self.key_string_value_json_storage_cls(
                namespace="llm_response_cache",
                global_config=asdict(self),
                embedding_func=None,
            )
            if self.enable_llm_cache
            else None
        )
        self.embedding_func = limit_async_func_call(self.embedding_func_max_async)(
            self.embedding_func
        )

        self.full_docs = self.key_string_value_json_storage_cls(
            namespace="full_docs",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
        )
        self.text_chunks = self.key_string_value_json_storage_cls(
            namespace="text_chunks",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
        )
        self.multimodal_chunks = self.key_string_value_json_storage_cls(
            namespace="multimodal_chunks",
            global_config=asdict(self),
            embedding_func=None,
        )
        self.chunk_entity_relation_graph = self.graph_storage_cls(
            namespace="chunk_entity_relation",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
        )

        self.entities_vdb = self.vector_db_storage_cls(
            namespace="entities",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
            meta_fields={"entity_name"},
        )
        self.hyperedges_vdb = self.vector_db_storage_cls(
            namespace="hyperedges",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
            meta_fields={"hyperedge_name"},
        )
        self.chunks_vdb = self.vector_db_storage_cls(
            namespace="chunks",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
        )

        self.llm_model_func = limit_async_func_call(self.llm_model_max_async)(
            partial(
                self.llm_model_func,
                hashing_kv=self.llm_response_cache,
                **self.llm_model_kwargs,
            )
        )

        if getattr(self, "twi_runner", None) is None:
            self.twi_runner = ThinkWithImageRunner()

    def _get_storage_class(self) -> Type[BaseGraphStorage]:
        return {
            # kv storage
            "JsonKVStorage": JsonKVStorage,
            "OracleKVStorage": OracleKVStorage,
            "MongoKVStorage": MongoKVStorage,
            "TiDBKVStorage": TiDBKVStorage,
            # vector storage
            "NanoVectorDBStorage": NanoVectorDBStorage,
            "OracleVectorDBStorage": OracleVectorDBStorage,
            "MilvusVectorDBStorge": MilvusVectorDBStorge,
            "ChromaVectorDBStorage": ChromaVectorDBStorage,
            "TiDBVectorDBStorage": TiDBVectorDBStorage,
            # graph storage
            "NetworkXStorage": NetworkXStorage,
            "Neo4JStorage": Neo4JStorage,
            "OracleGraphStorage": OracleGraphStorage,
            # "ArangoDBStorage": ArangoDBStorage
        }

    def insert(self, string_or_strings):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.ainsert(string_or_strings))

    async def ainsert(self, string_or_strings):
        update_storage = False
        try:
            normalized: List[Union[str, StructuredChunk]] = []
            if isinstance(string_or_strings, (str, StructuredChunk, dict)):
                normalized = [string_or_strings]
            elif isinstance(string_or_strings, Iterable):
                normalized = list(string_or_strings)
            else:
                raise TypeError(f"Unsupported insert payload: {type(string_or_strings)!r}")

            raw_docs: List[str] = []
            structured_chunks: List[StructuredChunk] = []
            for item in normalized:
                if isinstance(item, StructuredChunk):
                    structured_chunks.append(item.ensure_text())
                elif isinstance(item, dict):
                    structured_chunks.append(
                        StructuredChunk(
                            text=str(item.get("text", "")),
                            metadata=item.get("metadata", {}),
                        ).ensure_text()
                    )
                elif isinstance(item, str):
                    raw_docs.append(item)
                else:
                    raise TypeError(f"Unsupported insert item type: {type(item)!r}")

            new_docs: Dict[str, Dict[str, Any]] = {}
            inserting_chunks: Dict[str, Dict[str, Any]] = {}
            multimodal_chunk_records: Dict[str, Dict[str, Any]] = {}

            if raw_docs:
                candidate_docs = {
                    compute_mdhash_id(c.strip(), prefix="doc-"): {"content": c.strip()}
                    for c in raw_docs
                }
                allowed_doc_keys = await self.full_docs.filter_keys(list(candidate_docs.keys()))
                new_raw_docs = {k: candidate_docs[k] for k in allowed_doc_keys}
                if new_raw_docs:
                    update_storage = True
                    new_docs.update(new_raw_docs)
                    logger.info(f"[New Docs] inserting {len(new_raw_docs)} docs")

                    for doc_key, doc in tqdm_async(
                        new_raw_docs.items(), desc="Chunking documents", unit="doc"
                    ):
                        chunks = {
                            compute_mdhash_id(dp["content"], prefix="chunk-"): {
                                **dp,
                                "full_doc_id": doc_key,
                                "metadata": {
                                    "type": "text",
                                    "doc_id": doc_key,
                                    "chunk_order_index": dp.get("chunk_order_index"),
                                },
                            }
                            for dp in chunking_by_token_size(
                                doc["content"],
                                overlap_token_size=self.chunk_overlap_token_size,
                                max_token_size=self.chunk_token_size,
                                tiktoken_model=self.tiktoken_model_name,
                            )
                        }
                        inserting_chunks.update(chunks)
                else:
                    logger.info("All textual docs already exist in storage")

            if structured_chunks:
                doc_text_map: Dict[str, List[str]] = {}
                for chunk in structured_chunks:
                    metadata = dict(chunk.metadata or {})
                    metadata_payload = {
                        "text": chunk.text,
                        "metadata": metadata,
                    }
                    chunk_id = metadata.get("chunk_id") or compute_mdhash_id(
                        json.dumps(metadata_payload, sort_keys=True, ensure_ascii=False),
                        prefix="chunk-",
                    )
                    metadata.setdefault("chunk_id", chunk_id)
                    doc_id = metadata.get("doc_id") or compute_mdhash_id(
                        chunk.text, prefix="doc-"
                    )
                    metadata.setdefault("doc_id", doc_id)
                    doc_text_map.setdefault(doc_id, []).append(chunk.text)

                    chunk_entry = {
                        "content": chunk.text,
                        "metadata": metadata,
                        "full_doc_id": doc_id,
                    }
                    if "chunk_order_index" not in chunk_entry["metadata"]:
                        chunk_entry["metadata"]["chunk_order_index"] = len(
                            doc_text_map[doc_id]
                        ) - 1
                    inserting_chunks[chunk_id] = chunk_entry

                    if metadata.get("type") != "text":
                        multimodal_chunk_records[chunk_id] = {
                            "content": chunk.text,
                            "metadata": metadata,
                        }

                # Prepare aggregated doc entries for structured chunks
                if doc_text_map:
                    for doc_id, parts in doc_text_map.items():
                        doc_key = doc_id
                        aggregated = "\n\n".join(parts)
                        if doc_key not in new_docs:
                            new_docs[doc_key] = {"content": aggregated, "structured": True}
                        else:
                            new_docs[doc_key]["content"] = aggregated
                    update_storage = True

            if not inserting_chunks and not new_docs:
                logger.warning("No new content detected for insertion")
                return

            if inserting_chunks:
                allowed_chunk_keys = await self.text_chunks.filter_keys(list(inserting_chunks.keys()))
                inserting_chunks = {
                    k: inserting_chunks[k] for k in allowed_chunk_keys if k in inserting_chunks
                }
                if multimodal_chunk_records:
                    multimodal_chunk_records = {
                        k: multimodal_chunk_records[k]
                        for k in allowed_chunk_keys
                        if k in multimodal_chunk_records
                    }

            if not inserting_chunks:
                logger.info("All chunks already exist in storage; skipping chunk insertion")
            else:
                update_storage = True
                logger.info(f"[New Chunks] inserting {len(inserting_chunks)} chunks")
                await self.chunks_vdb.upsert(inserting_chunks)

                logger.info("[Entity Extraction]...")
                maybe_new_kg = await extract_entities(
                    inserting_chunks,
                    knowledge_graph_inst=self.chunk_entity_relation_graph,
                    entity_vdb=self.entities_vdb,
                    hyperedge_vdb=self.hyperedges_vdb,
                    global_config=asdict(self),
                )
                if maybe_new_kg is None:
                    logger.warning("No new hyperedges and entities found")
                else:
                    self.chunk_entity_relation_graph = maybe_new_kg

                await self.text_chunks.upsert(inserting_chunks)
                if multimodal_chunk_records:
                    await self.multimodal_chunks.upsert(multimodal_chunk_records)

            if new_docs:
                await self.full_docs.upsert(new_docs)
        finally:
            if update_storage:
                await self._insert_done()

    async def _insert_done(self):
        tasks = []
        for storage_inst in [
            self.full_docs,
            self.text_chunks,
            self.multimodal_chunks,
            self.llm_response_cache,
            self.entities_vdb,
            self.hyperedges_vdb,
            self.chunks_vdb,
            self.chunk_entity_relation_graph,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)

    def insert_custom_kg(self, custom_kg: dict):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.ainsert_custom_kg(custom_kg))

    async def ainsert_custom_kg(self, custom_kg: dict):
        update_storage = False
        try:
            # Insert chunks into vector storage
            all_chunks_data = {}
            chunk_to_source_map = {}
            for chunk_data in custom_kg.get("chunks", []):
                chunk_content = chunk_data["content"]
                source_id = chunk_data["source_id"]
                chunk_id = compute_mdhash_id(chunk_content.strip(), prefix="chunk-")

                chunk_entry = {"content": chunk_content.strip(), "source_id": source_id}
                all_chunks_data[chunk_id] = chunk_entry
                chunk_to_source_map[source_id] = chunk_id
                update_storage = True

            if self.chunks_vdb is not None and all_chunks_data:
                await self.chunks_vdb.upsert(all_chunks_data)
            if self.text_chunks is not None and all_chunks_data:
                await self.text_chunks.upsert(all_chunks_data)

            # Insert entities into knowledge graph
            all_entities_data = []
            for entity_data in custom_kg.get("entities", []):
                entity_name = f'"{entity_data["entity_name"].upper()}"'
                entity_type = entity_data.get("entity_type", "UNKNOWN")
                description = entity_data.get("description", "No description provided")
                # source_id = entity_data["source_id"]
                source_chunk_id = entity_data.get("source_id", "UNKNOWN")
                source_id = chunk_to_source_map.get(source_chunk_id, "UNKNOWN")

                # Log if source_id is UNKNOWN
                if source_id == "UNKNOWN":
                    logger.warning(
                        f"Entity '{entity_name}' has an UNKNOWN source_id. Please check the source mapping."
                    )

                # Prepare node data
                node_data = {
                    "entity_type": entity_type,
                    "description": description,
                    "source_id": source_id,
                }
                # Insert node data into the knowledge graph
                await self.chunk_entity_relation_graph.upsert_node(
                    entity_name, node_data=node_data
                )
                node_data["entity_name"] = entity_name
                all_entities_data.append(node_data)
                update_storage = True

            # Insert relationships into knowledge graph
            all_relationships_data = []
            for relationship_data in custom_kg.get("relationships", []):
                src_id = f'"{relationship_data["src_id"].upper()}"'
                tgt_id = f'"{relationship_data["tgt_id"].upper()}"'
                description = relationship_data["description"]
                keywords = relationship_data["keywords"]
                weight = relationship_data.get("weight", 1.0)
                # source_id = relationship_data["source_id"]
                source_chunk_id = relationship_data.get("source_id", "UNKNOWN")
                source_id = chunk_to_source_map.get(source_chunk_id, "UNKNOWN")

                # Log if source_id is UNKNOWN
                if source_id == "UNKNOWN":
                    logger.warning(
                        f"Relationship from '{src_id}' to '{tgt_id}' has an UNKNOWN source_id. Please check the source mapping."
                    )

                # Check if nodes exist in the knowledge graph
                for need_insert_id in [src_id, tgt_id]:
                    if not (
                        await self.chunk_entity_relation_graph.has_node(need_insert_id)
                    ):
                        await self.chunk_entity_relation_graph.upsert_node(
                            need_insert_id,
                            node_data={
                                "source_id": source_id,
                                "description": "UNKNOWN",
                                "entity_type": "UNKNOWN",
                            },
                        )

                # Insert edge into the knowledge graph
                await self.chunk_entity_relation_graph.upsert_edge(
                    src_id,
                    tgt_id,
                    edge_data={
                        "weight": weight,
                        "description": description,
                        "keywords": keywords,
                        "source_id": source_id,
                    },
                )
                edge_data = {
                    "src_id": src_id,
                    "tgt_id": tgt_id,
                    "description": description,
                    "keywords": keywords,
                }
                all_relationships_data.append(edge_data)
                update_storage = True

            # Insert entities into vector storage if needed
            if self.entities_vdb is not None:
                data_for_vdb = {
                    compute_mdhash_id(dp["entity_name"], prefix="ent-"): {
                        "content": dp["entity_name"] + dp["description"],
                        "entity_name": dp["entity_name"],
                    }
                    for dp in all_entities_data
                }
                await self.entities_vdb.upsert(data_for_vdb)

            # Insert relationships into vector storage if needed
            if self.hyperedges_vdb is not None:
                data_for_vdb = {
                    compute_mdhash_id(dp["src_id"] + dp["tgt_id"], prefix="rel-"): {
                        "src_id": dp["src_id"],
                        "tgt_id": dp["tgt_id"],
                        "content": dp["keywords"]
                        + dp["src_id"]
                        + dp["tgt_id"]
                        + dp["description"],
                    }
                    for dp in all_relationships_data
                }
                await self.hyperedges_vdb.upsert(data_for_vdb)
        finally:
            if update_storage:
                await self._insert_done()

    def query(self, query: str, param: Optional[QueryParam] = None):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.aquery(query, param))

    async def aquery(self, query: str, param: Optional[QueryParam] = None):
        if param is None:
            param = QueryParam()
        else:
            param = replace(param)
        kg_result = await kg_query(
            query,
            self.chunk_entity_relation_graph,
            self.entities_vdb,
            self.hyperedges_vdb,
            self.text_chunks,
            param,
            asdict(self),
            hashing_kv=self.llm_response_cache,
        )
        response_text = kg_result.get("response")
        modal_chunks = kg_result.get("modal_chunks") or []

        vlm_answer = None
        if self.vlm_client and modal_chunks:
            vlm_answer = await self._run_vlm(
                query=query,
                modal_chunks=modal_chunks,
                context=kg_result.get("context"),
            )

        final_answer = self._merge_answers(response_text, vlm_answer)

        await self._query_done()
        return final_answer

    async def _run_vlm(
        self,
        *,
        query: str,
        modal_chunks: List[Dict[str, Any]],
        context: Optional[str] = None,
    ) -> Optional[str]:
        if not modal_chunks or self.vlm_client is None:
            return None

        images: List[str] = []
        details: List[str] = []
        image_metadata_map: Dict[str, List[Dict[str, Any]]] = {}
        for idx, chunk in enumerate(modal_chunks, start=1):
            metadata = chunk.get("metadata", {}) or {}
            caption = metadata.get("caption") or chunk.get("content") or ""
            image_path = metadata.get("image_path")
            doc_id = metadata.get("doc_id")
            details.append(
                f"{idx}. 文档ID: {doc_id or '未知'}, 类型: {metadata.get('type', 'modal')}, 描述: {caption.strip()}"
            )
            if image_path:
                images.append(image_path)
                image_metadata_map.setdefault(image_path, []).append(metadata)
        if not images:
            logger.warning("VLM invocation skipped: no valid image paths found")
            return None

        twi_blocks: List[str] = []
        if getattr(self, "twi_runner", None) is not None:
            try:
                twi_results = await self.twi_runner.arun(query, images)
            except Exception as exc:  # pragma: no cover - logging only
                logger.warning("TWI runner failed, fallback to direct VLM: %s", exc)
                twi_results = []
            for result in twi_results or []:
                twi_blocks.append(result.to_prompt_block())
                for meta in image_metadata_map.get(result.image_path, []):
                    meta["twi_summary"] = result.summary
                    meta["twi_search_target"] = result.search_target
                    meta["twi_detections"] = result.detections

        prompt_parts = [
            "你是一名多模态检索问答助手。",
            f"用户问题：{query}",
            "以下是与问题相关的多模态片段，请综合这些信息回答：",
            "\n".join(details),
        ]
        if twi_blocks:
            prompt_parts.append("[Think-With-Image 摘要]")
            prompt_parts.append("\n\n".join(twi_blocks))
        if context:
            prompt_parts.append("文本检索上下文：")
            prompt_parts.append(context)
        prompt_parts.append("请结合图像内容和文本上下文，给出准确的回答。")
        prompt = "\n".join(prompt_parts)

        try:
            vlm_answer = await self.vlm_client.generate(
                prompt,
                images=images,
                max_tokens=self.addon_params.get("vlm_max_tokens", 512),
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(f"VLM 生成失败: {exc}")
            return None
        return vlm_answer.strip()

    def _merge_answers(
        self, text_answer: Optional[str], vlm_answer: Optional[str]
    ) -> str:
        text_answer = (text_answer or "").strip()
        vlm_answer = (vlm_answer or "").strip()

        if vlm_answer and (not text_answer or text_answer == PROMPTS["fail_response"]):
            return vlm_answer
        if text_answer and vlm_answer:
            return f"{text_answer}\n\n[视觉补充]\n{vlm_answer}"
        if text_answer:
            return text_answer
        if vlm_answer:
            return vlm_answer
        return PROMPTS["fail_response"]

    async def _query_done(self):
        tasks = []
        for storage_inst in [self.llm_response_cache]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)

    def delete_by_entity(self, entity_name: str):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.adelete_by_entity(entity_name))

    async def adelete_by_entity(self, entity_name: str):
        entity_name = f'"{entity_name.upper()}"'

        try:
            await self.entities_vdb.delete_entity(entity_name)
            await self.hyperedges_vdb.delete_relation(entity_name)
            await self.chunk_entity_relation_graph.delete_node(entity_name)

            logger.info(
                f"Entity '{entity_name}' and its relationships have been deleted."
            )
            await self._delete_by_entity_done()
        except Exception as e:
            logger.error(f"Error while deleting entity '{entity_name}': {e}")

    async def _delete_by_entity_done(self):
        tasks = []
        for storage_inst in [
            self.entities_vdb,
            self.hyperedges_vdb,
            self.chunk_entity_relation_graph,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)
