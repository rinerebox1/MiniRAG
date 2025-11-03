import asyncio
import json
import re
from typing import Union
from collections import Counter, defaultdict
import warnings
import json_repair
from datetime import datetime

from .utils import (
    list_of_list_to_csv,
    truncate_list_by_token_size,
    split_string_by_multi_markers,
    logger,
    locate_json_string_body_from_string,
    process_combine_contexts,
    clean_str,
    edge_vote_path,
    encode_string_by_tiktoken,
    decode_tokens_by_tiktoken,
    is_float_regex,
    pack_user_ass_to_openai_messages,
    compute_mdhash_id,
    calculate_similarity,
    cal_path_score_list,
)
from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    TextChunkSchema,
    QueryParam,
)
from .prompt import GRAPH_FIELD_SEP, PROMPTS


def _parse_metadata_dict(raw_metadata):
    if isinstance(raw_metadata, dict):
        return raw_metadata
    if isinstance(raw_metadata, str):
        try:
            parsed = json.loads(raw_metadata)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    return {}


def _extract_category_from_metadata(raw_metadata):
    meta = _parse_metadata_dict(raw_metadata)
    return meta.get("category")


def _debug_log_filter_samples(context_label: str, phase_label: str, units: list, metadata_filter: dict):
    try:
        max_samples = 5
        samples = []
        for u in (units or [])[:max_samples]:
            cid = u.get("id", "")
            cat = _extract_category_from_metadata(u.get("metadata"))
            dist = u.get("_distance", "N/A")
            samples.append(f"- id={str(cid)[:16]}..., category={cat}, distance={dist}")
        print(f"[{context_label}] Metadata filter: {metadata_filter}")
        print(f"[{context_label}] {phase_label}: {len(units or [])} candidates")
        if samples:
            print(f"[{context_label}] {phase_label} samples:\n  " + "\n  ".join(samples))
    except Exception as e:
        print(f"[{context_label}] Debug logging failed: {e}")


def chunking_by_token_size(
    content: str, overlap_token_size=128, max_token_size=1024, tiktoken_model="gpt-4o"
):
    tokens = encode_string_by_tiktoken(content, model_name=tiktoken_model)
    results = []
    for index, start in enumerate(
        range(0, len(tokens), max_token_size - overlap_token_size)
    ):
        chunk_content = decode_tokens_by_tiktoken(
            tokens[start : start + max_token_size], model_name=tiktoken_model
        )
        results.append(
            {
                "tokens": min(max_token_size, len(tokens) - start),
                "content": chunk_content.strip(),
                "chunk_order_index": index,
            }
        )
    return results


async def _handle_single_entity_extraction(
    record_attributes: list[str],
    chunk_key: str,
):
    if len(record_attributes) < 4 or record_attributes[0] != '"entity"':
        return None
    # add this record as a node in the G
    entity_name = clean_str(record_attributes[1].upper())
    if not entity_name.strip():
        return None
    entity_type = clean_str(record_attributes[2].upper())
    entity_description = clean_str(record_attributes[3])
    entity_source_id = chunk_key
    return dict(
        entity_name=entity_name,
        entity_type=entity_type,
        description=entity_description,
        source_id=entity_source_id,
    )


async def _handle_single_relationship_extraction(
    record_attributes: list[str],
    chunk_key: str,
):
    if len(record_attributes) < 5 or record_attributes[0] != '"relationship"':
        return None
    # add this record as edge
    source = clean_str(record_attributes[1].upper())
    target = clean_str(record_attributes[2].upper())
    edge_description = clean_str(record_attributes[3])

    edge_keywords = clean_str(record_attributes[4])
    edge_source_id = chunk_key
    weight = (
        float(record_attributes[-1]) if is_float_regex(record_attributes[-1]) else 1.0
    )
    return dict(
        src_id=source,
        tgt_id=target,
        weight=weight,
        description=edge_description,
        keywords=edge_keywords,
        source_id=edge_source_id,
    )


async def _merge_nodes_then_upsert(
    entity_name: str,
    nodes_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    already_entitiy_types = []
    already_source_ids = []
    already_description = []

    already_node = await knowledge_graph_inst.get_node(entity_name)
    if already_node is not None:
        already_entitiy_types.append(already_node["entity_type"])
        already_source_ids.extend(
            split_string_by_multi_markers(already_node["source_id"], [GRAPH_FIELD_SEP])
        )
        already_description.append(already_node["description"])

    entity_type = sorted(
        Counter(
            [dp["entity_type"] for dp in nodes_data] + already_entitiy_types
        ).items(),
        key=lambda x: x[1],
        reverse=True,
    )[0][0]

    description = GRAPH_FIELD_SEP.join(
        sorted(set([dp["description"] for dp in nodes_data] + already_description))
    )
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in nodes_data] + already_source_ids)
    )

    # description = await _handle_entity_relation_summary(
    #     entity_name, description, global_config
    # )
    node_data = dict(
        entity_type=entity_type,
        description=description,
        source_id=source_id,
    )
    await knowledge_graph_inst.upsert_node(
        entity_name,
        node_data=node_data,
    )
    node_data["entity_name"] = entity_name
    return node_data


async def _merge_edges_then_upsert(
    src_id: str,
    tgt_id: str,
    edges_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    # Safely get existing edge and merge properties
    already_edge = await knowledge_graph_inst.get_edge(src_id, tgt_id) or {}

    # Combine attributes, giving precedence to new data
    weight = sum(dp["weight"] for dp in edges_data) + already_edge.get("weight", 0.0)

    description = GRAPH_FIELD_SEP.join(sorted(set(
        [dp["description"] for dp in edges_data] +
        split_string_by_multi_markers(already_edge.get("description", ""), [GRAPH_FIELD_SEP])
    )))

    keywords = GRAPH_FIELD_SEP.join(sorted(set(
        [dp["keywords"] for dp in edges_data] +
        split_string_by_multi_markers(already_edge.get("keywords", ""), [GRAPH_FIELD_SEP])
    )))

    source_id = GRAPH_FIELD_SEP.join(sorted(set(
        [dp["source_id"] for dp in edges_data] +
        split_string_by_multi_markers(already_edge.get("source_id", ""), [GRAPH_FIELD_SEP])
    )))

    # Ensure source and target nodes exist
    for node_id in [src_id, tgt_id]:
        if not await knowledge_graph_inst.has_node(node_id):
            await knowledge_graph_inst.upsert_node(
                node_id,
                node_data={
                    "source_id": source_id,
                    "description": "Auto-generated node.",
                    "entity_type": '"UNKNOWN"',
                },
            )

    edge_data = dict(
        weight=weight,
        description=description,
        keywords=keywords,
        source_id=source_id,
    )

    await knowledge_graph_inst.upsert_edge(src_id, tgt_id, edge_data=edge_data)

    return {"src_id": src_id, "tgt_id": tgt_id, **edge_data}


async def extract_entities(
    chunks: dict[str, TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
    entity_vdb: BaseVectorStorage,
    entity_name_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    global_config: dict,
) -> Union[BaseGraphStorage, None]:
    use_llm_func: callable = global_config["llm_model_func"]
    entity_extract_max_gleaning = global_config["entity_extract_max_gleaning"]

    ordered_chunks = list(chunks.items())
    # if global_config['RAGmode'] == 'minirag':
    #     # entity_extract_prompt = PROMPTS["entity_extraction_noDes"]
    #     entity_extract_prompt = PROMPTS["entity_extraction"]
    # else:
    entity_extract_prompt = PROMPTS["entity_extraction"]

    context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=",".join(PROMPTS["DEFAULT_ENTITY_TYPES"]),
    )
    continue_prompt = PROMPTS["entiti_continue_extraction"]

    if_loop_prompt = PROMPTS["entiti_if_loop_extraction"]

    already_processed = 0
    already_entities = 0
    already_relations = 0

    async def _process_single_content(chunk_key_dp: tuple[str, TextChunkSchema]):
        nonlocal already_processed, already_entities, already_relations
        chunk_key = chunk_key_dp[0]
        chunk_dp = chunk_key_dp[1]
        content = chunk_dp["content"]
        hint_prompt = entity_extract_prompt.format(**context_base, input_text=content)
        final_result = await use_llm_func(hint_prompt)

        history = pack_user_ass_to_openai_messages(hint_prompt, final_result)
        for now_glean_index in range(entity_extract_max_gleaning):
            glean_result = await use_llm_func(continue_prompt, history_messages=history)

            history += pack_user_ass_to_openai_messages(continue_prompt, glean_result)
            final_result += glean_result
            if now_glean_index == entity_extract_max_gleaning - 1:
                break

            if_loop_result: str = await use_llm_func(
                if_loop_prompt, history_messages=history
            )
            if_loop_result = if_loop_result.strip().strip('"').strip("'").lower()
            if if_loop_result != "yes":
                break

        records = split_string_by_multi_markers(
            final_result,
            [context_base["record_delimiter"], context_base["completion_delimiter"]],
        )

        maybe_nodes = defaultdict(list)
        maybe_edges = defaultdict(list)
        for record in records:
            record = re.search(r"\((.*)\)", record)
            if record is None:
                continue
            record = record.group(1)
            record_attributes = split_string_by_multi_markers(
                record, [context_base["tuple_delimiter"]]
            )
            if_entities = await _handle_single_entity_extraction(
                record_attributes, chunk_key
            )
            if if_entities is not None:
                maybe_nodes[if_entities["entity_name"]].append(if_entities)
                continue

            if_relation = await _handle_single_relationship_extraction(
                record_attributes, chunk_key
            )
            if if_relation is not None:
                maybe_edges[(if_relation["src_id"], if_relation["tgt_id"])].append(
                    if_relation
                )
        already_processed += 1
        already_entities += len(maybe_nodes)
        already_relations += len(maybe_edges)
        now_ticks = PROMPTS["process_tickers"][
            already_processed % len(PROMPTS["process_tickers"])
        ]
        print(
            f"{now_ticks} Processed {already_processed} chunks, {already_entities} entities(duplicated), {already_relations} relations(duplicated)\r",
            end="",
            flush=True,
        )
        return dict(maybe_nodes), dict(maybe_edges)

    # use_llm_func is wrapped in ascynio.Semaphore, limiting max_async callings
    results = await asyncio.gather(
        *[_process_single_content(c) for c in ordered_chunks]
    )
    print()  # clear the progress bar
    maybe_nodes = defaultdict(list)
    maybe_edges = defaultdict(list)
    for m_nodes, m_edges in results:
        for k, v in m_nodes.items():
            maybe_nodes[k].extend(v)
        for k, v in m_edges.items():
            maybe_edges[tuple(sorted(k))].extend(v)
    all_entities_data = await asyncio.gather(
        *[
            _merge_nodes_then_upsert(k, v, knowledge_graph_inst, global_config)
            for k, v in maybe_nodes.items()
        ]
    )
    all_relationships_data = await asyncio.gather(
        *[
            _merge_edges_then_upsert(k[0], k[1], v, knowledge_graph_inst, global_config)
            for k, v in maybe_edges.items()
        ]
    )
    if not len(all_entities_data):
        logger.warning("Didn't extract any entities, maybe your LLM is not working")
        return None
    if not len(all_relationships_data):
        logger.warning(
            "Didn't extract any relationships, maybe your LLM is not working"
        )
        return None

    if entity_vdb is not None:
        data_for_vdb = {
            compute_mdhash_id(dp["entity_name"], prefix="ent-"): {
                "content": dp["entity_name"] + dp["description"],
                "entity_name": dp["entity_name"],
                "metadata": chunks.get(dp["source_id"].split(GRAPH_FIELD_SEP)[0], {}).get("metadata", {}),
            }
            for dp in all_entities_data
        }
        await entity_vdb.upsert(data_for_vdb)
    if entity_vdb is not None:
        data_for_vdb = {
            compute_mdhash_id(dp["entity_name"], prefix="ent-"): {
                "content": dp["entity_name"] + " " + dp["description"],
                "entity_name": dp["entity_name"],
                "metadata": chunks.get(dp["source_id"].split(GRAPH_FIELD_SEP)[0], {}).get("metadata", {}),
            }
            for dp in all_entities_data
        }
        await entity_vdb.upsert(data_for_vdb)

    if entity_name_vdb is not None:
        # 🆕 メタデータを付与: source_id の最初のチャンクから取得
        # これにより、minirag_query での metadata_filter が機能し、検索性能にも影響する
        data_for_vdb = {
            compute_mdhash_id(dp["entity_name"], prefix="Ename-"): {
                "content": dp["entity_name"],
                "entity_name": dp["entity_name"],
                "metadata": chunks.get(dp["source_id"].split(GRAPH_FIELD_SEP)[0], {}).get("metadata", {}),
            }
            for dp in all_entities_data
        }
        await entity_name_vdb.upsert(data_for_vdb)

    if relationships_vdb is not None:
        data_for_vdb = {
            compute_mdhash_id(dp["src_id"] + dp["tgt_id"], prefix="rel-"): {
                "src_id": dp["src_id"],
                "tgt_id": dp["tgt_id"],
                "content": dp["keywords"]
                + " " + dp["src_id"]
                + " " + dp["tgt_id"]
                + " " + dp["description"],
                "metadata": chunks.get(dp["source_id"].split(GRAPH_FIELD_SEP)[0], {}).get("metadata", {}),
            }
            for dp in all_relationships_data
        }

        await relationships_vdb.upsert(data_for_vdb)

    return knowledge_graph_inst


async def local_query(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    chunks_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
) -> str:
    context = None
    use_model_func = global_config["llm_model_func"]

    kw_prompt_temp = PROMPTS["keywords_extraction"]
    kw_prompt = kw_prompt_temp.format(query=query)
    result = await use_model_func(kw_prompt)
    json_text = locate_json_string_body_from_string(result)

    try:
        keywords_data = json.loads(json_text)
        keywords = keywords_data.get("low_level_keywords", [])
        keywords = ", ".join(keywords)
    except json.JSONDecodeError:
        try:
            result = (
                result.replace(kw_prompt[:-1], "")
                .replace("user", "")
                .replace("model", "")
                .strip()
            )
            result = "{" + result.split("{")[1].split("}")[0] + "}"

            keywords_data = json.loads(result)
            keywords = keywords_data.get("low_level_keywords", [])
            keywords = ", ".join(keywords)
        # Handle parsing error
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            return PROMPTS["fail_response"]
    if keywords:
        context, source = await _build_local_query_context(
            keywords,
            knowledge_graph_inst,
            entities_vdb,
            chunks_vdb,
            text_chunks_db,
            query_param,
        )
        # Noneが返された場合の処理
        if context is None:
            context, source = None, []
    if query_param.only_need_context:
        return context
    if context is None:
        return PROMPTS["fail_response"]
    sys_prompt_temp = PROMPTS["rag_response"]
    sys_prompt = sys_prompt_temp.format(
        context_data=context, response_type=query_param.response_type
    )
    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
    )
    if len(response) > len(sys_prompt):
        response = (
            response.replace(sys_prompt, "")
            .replace("user", "")
            .replace("model", "")
            .replace(query, "")
            .replace("<system>", "")
            .replace("</system>", "")
            .strip()
        )

    return response


async def _build_local_query_context(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    chunks_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
):
    results = await entities_vdb.query(
        query,
        top_k=query_param.top_k,
        # metadata_filter=query_param.metadata_filter, # メタデータでの絞り込みはチャンク取得後に行う
        start_time=query_param.start_time,
        end_time=query_param.end_time,
    )

    if not len(results):
        return None, []
    node_datas = await asyncio.gather(
        *[knowledge_graph_inst.get_node(r["entity_name"]) for r in results]
    )
    if not all([n is not None for n in node_datas]):
        logger.warning("Some nodes are missing, maybe the storage is damaged")
    node_degrees = await asyncio.gather(
        *[knowledge_graph_inst.node_degree(r["entity_name"]) for r in results]
    )
    node_datas = [
        {**n, "entity_name": k["entity_name"], "rank": d}
        for k, n, d in zip(results, node_datas, node_degrees)
        if n is not None
    ]  # what is this text_chunks_db doing.  dont remember it in airvx.  check the diagram.
    use_text_units = await _find_most_related_text_unit_from_entities(
        node_datas, query_param, text_chunks_db, knowledge_graph_inst
    )

    # 取得したチャンクの距離を一括取得
    if use_text_units:
        chunk_ids_set = {u["id"] for u in use_text_units}
        results_dist = await chunks_vdb.query(query, top_k=len(chunk_ids_set) * 2, debug=False)
        dist_lookup = {r["id"]: r.get("distance") for r in results_dist if r["id"] in chunk_ids_set}
        for u in use_text_units:
            u["_distance"] = dist_lookup.get(u["id"], "N/A")

    # 取得したチャンクに対してメタデータフィルタを適用
    if query_param.metadata_filter:
        # フィルタ適用前の件数を記録
        _before_cnt = len(use_text_units)
        _debug_log_filter_samples("Local", "before", use_text_units, query_param.metadata_filter)
        filtered_text_units = []
        for unit in use_text_units:
            chunk_metadata_raw = unit.get("metadata")
            chunk_metadata = {}
            if isinstance(chunk_metadata_raw, dict):
                chunk_metadata = chunk_metadata_raw
            elif isinstance(chunk_metadata_raw, str):
                try:
                    # 文字列の場合、JSONとしてパースを試みる
                    parsed_meta = json.loads(chunk_metadata_raw)
                    if isinstance(parsed_meta, dict):
                        chunk_metadata = parsed_meta
                except json.JSONDecodeError:
                    pass  # パース失敗時は空の辞書として扱う

            is_match = True
            for key, value in query_param.metadata_filter.items():
                if chunk_metadata.get(key) != value:
                    is_match = False
                    break
            if is_match:
                filtered_text_units.append(unit)
        use_text_units = filtered_text_units
        print(f"[Local] filter reduced candidates: before={_before_cnt} -> after={len(use_text_units)}")
        _debug_log_filter_samples("Local", "after", use_text_units, query_param.metadata_filter)

    # フィルタリング後のチャンクに基づいてエンティティの情報を再構築する
    filtered_chunk_content_map = {unit["id"]: unit["content"] for unit in use_text_units}
    final_node_datas = []
    for node in node_datas:
        source_ids = split_string_by_multi_markers(node.get("source_id", ""), [GRAPH_FIELD_SEP])
        relevant_content = [
            filtered_chunk_content_map[sid] for sid in source_ids if sid in filtered_chunk_content_map
        ]
        
        if relevant_content:
            node["description"] = GRAPH_FIELD_SEP.join(relevant_content)
            final_node_datas.append(node)
    node_datas = final_node_datas

    use_relations = await _find_most_related_edges_from_entities(
        node_datas, query_param, knowledge_graph_inst
    )
    logger.info(
        f"Local query uses {len(node_datas)} entites, {len(use_relations)} relations, {len(use_text_units)} text units"
    )
    entites_section_list = [["id", "entity", "type", "description", "rank"]]
    for i, n in enumerate(node_datas):
        entites_section_list.append(
            [
                i,
                n["entity_name"],
                n.get("entity_type", "UNKNOWN"),
                n.get("description", "UNKNOWN"),
                n["rank"],
            ]
        )
    entities_context = list_of_list_to_csv(entites_section_list)

    relations_section_list = [
        ["id", "source", "target", "description", "keywords", "weight", "rank"]
    ]
    for i, e in enumerate(use_relations):
        relations_section_list.append(
            [
                i,
                e["src_tgt"][0],
                e["src_tgt"][1],
                e["description"],
                e["keywords"],
                e["weight"],
                e["rank"],
            ]
        )
    relations_context = list_of_list_to_csv(relations_section_list)

    text_units_section_list = [["id", "content"]]
    source = []  # sourceリストを追加
    for i, t in enumerate(use_text_units):
        text_units_section_list.append([i, t["content"]])
        source.append(t["content"])  # sourceリストに追加
    text_units_context = list_of_list_to_csv(text_units_section_list)
    context = f"""
-----Entities-----
```csv
{entities_context}
```
-----Relationships-----
```csv
{relations_context}
```
-----Sources-----
```csv
{text_units_context}
```
"""
    return context, source  # contextとsourceの両方を返す


async def _find_most_related_text_unit_from_entities(
    node_datas: list[dict],
    query_param: QueryParam,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
):
    text_units = [
        split_string_by_multi_markers(dp["source_id"], [GRAPH_FIELD_SEP])
        for dp in node_datas
    ]
    edges = await asyncio.gather(
        *[knowledge_graph_inst.get_node_edges(dp["entity_name"]) for dp in node_datas]
    )
    all_one_hop_nodes = set()
    for this_edges in edges:
        if not this_edges:
            continue
        all_one_hop_nodes.update([e[1] for e in this_edges])

    all_one_hop_nodes = list(all_one_hop_nodes)
    all_one_hop_nodes_data = await asyncio.gather(
        *[knowledge_graph_inst.get_node(e) for e in all_one_hop_nodes]
    )

    # Add null check for node data
    all_one_hop_text_units_lookup = {
        k: set(split_string_by_multi_markers(v["source_id"], [GRAPH_FIELD_SEP]))
        for k, v in zip(all_one_hop_nodes, all_one_hop_nodes_data)
        if v is not None and "source_id" in v  # Add source_id check
    }

    all_text_units_lookup = {}
    for index, (this_text_units, this_edges) in enumerate(zip(text_units, edges)):
        for c_id in this_text_units:
            if c_id in all_text_units_lookup:
                continue
            relation_counts = 0
            if this_edges:  # Add check for None edges
                for e in this_edges:
                    if (
                        e[1] in all_one_hop_text_units_lookup
                        and c_id in all_one_hop_text_units_lookup[e[1]]
                    ):
                        relation_counts += 1

            chunk_data = await text_chunks_db.get_by_id(c_id)
            if chunk_data is not None and "content" in chunk_data:  # Add content check
                all_text_units_lookup[c_id] = {
                    "data": chunk_data,
                    "order": index,
                    "relation_counts": relation_counts,
                }

    # Filter out None values and ensure data has content
    all_text_units = [
        {"id": k, **v}
        for k, v in all_text_units_lookup.items()
        if v is not None and v.get("data") is not None and "content" in v["data"]
    ]

    if not all_text_units:
        logger.warning("No valid text units found")
        return []

    all_text_units = sorted(
        all_text_units, key=lambda x: (x["order"], -x["relation_counts"])
    )

    all_text_units = truncate_list_by_token_size(
        all_text_units,
        key=lambda x: x["data"]["content"],
        max_token_size=query_param.max_token_for_text_unit,
    )

    all_text_units = [t["data"] for t in all_text_units]

    # 時間フィルタを適用
    if query_param.start_time or query_param.end_time:
        filtered_by_time = []
        start_time_dt = datetime.fromisoformat(query_param.start_time) if query_param.start_time else None
        end_time_dt = datetime.fromisoformat(query_param.end_time) if query_param.end_time else None

        for unit in all_text_units:
            unit_time = unit.get("updated_at")
            if not unit_time:
                continue
            
            # JSONB から読み戻された場合、datetime は文字列になっている可能性がある
            if isinstance(unit_time, str):
                unit_time = datetime.fromisoformat(unit_time.replace('Z', '+00:00'))

            if start_time_dt and unit_time < start_time_dt:
                continue
            if end_time_dt and unit_time > end_time_dt:
                continue
            filtered_by_time.append(unit)
        all_text_units = filtered_by_time

    return all_text_units


async def _find_most_related_edges_from_entities(
    node_datas: list[dict],
    query_param: QueryParam,
    knowledge_graph_inst: BaseGraphStorage,
):
    all_related_edges = await asyncio.gather(
        *[knowledge_graph_inst.get_node_edges(dp["entity_name"]) for dp in node_datas]
    )
    all_edges = set()
    for this_edges in all_related_edges:
        all_edges.update([tuple(sorted(e)) for e in this_edges])
    all_edges = list(all_edges)
    all_edges_pack = await asyncio.gather(
        *[knowledge_graph_inst.get_edge(e[0], e[1]) for e in all_edges]
    )
    all_edges_degree = await asyncio.gather(
        *[knowledge_graph_inst.edge_degree(e[0], e[1]) for e in all_edges]
    )
    all_edges_data = [
        {"src_tgt": k, "rank": d, **v}
        for k, v, d in zip(all_edges, all_edges_pack, all_edges_degree)
        if v is not None
    ]
    all_edges_data = sorted(
        all_edges_data, key=lambda x: (x["rank"], x["weight"]), reverse=True
    )
    all_edges_data = truncate_list_by_token_size(
        all_edges_data,
        key=lambda x: x["description"],
        max_token_size=query_param.max_token_for_global_context,
    )
    return all_edges_data


async def global_query(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    chunks_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
) -> str:
    context = None
    use_model_func = global_config["llm_model_func"]

    kw_prompt_temp = PROMPTS["keywords_extraction"]
    kw_prompt = kw_prompt_temp.format(query=query)
    result = await use_model_func(kw_prompt)
    json_text = locate_json_string_body_from_string(result)

    try:
        keywords_data = json.loads(json_text)
        keywords = keywords_data.get("high_level_keywords", [])
        keywords = ", ".join(keywords)
    except json.JSONDecodeError:
        try:
            result = (
                result.replace(kw_prompt[:-1], "")
                .replace("user", "")
                .replace("model", "")
                .strip()
            )
            result = "{" + result.split("{")[1].split("}")[0] + "}"

            keywords_data = json.loads(result)
            keywords = keywords_data.get("high_level_keywords", [])
            keywords = ", ".join(keywords)

        except json.JSONDecodeError as e:
            # Handle parsing error
            print(f"JSON parsing error: {e}")
            return PROMPTS["fail_response"]
    if keywords:
        context, source = await _build_global_query_context(
            keywords,
            knowledge_graph_inst,
            entities_vdb,
            relationships_vdb,
            chunks_vdb,
            text_chunks_db,
            query_param,
        )
        # Noneが返された場合の処理
        if context is None:
            context, source = None, []

    if query_param.only_need_context:
        return context
    if context is None:
        return PROMPTS["fail_response"]

    sys_prompt_temp = PROMPTS["rag_response"]
    sys_prompt = sys_prompt_temp.format(
        context_data=context, response_type=query_param.response_type
    )
    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
    )
    if len(response) > len(sys_prompt):
        response = (
            response.replace(sys_prompt, "")
            .replace("user", "")
            .replace("model", "")
            .replace(query, "")
            .replace("<system>", "")
            .replace("</system>", "")
            .strip()
        )

    return response


async def _build_global_query_context(
    keywords,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    chunks_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
):
    results = await relationships_vdb.query(
        keywords,
        top_k=query_param.top_k,
        # metadata_filter=query_param.metadata_filter, # メタデータでの絞り込みはチャンク取得後に行う
        start_time=query_param.start_time,
        end_time=query_param.end_time,
    )

    if not len(results):
        return None, []

    edge_datas = await asyncio.gather(
        *[knowledge_graph_inst.get_edge(r["src_id"], r["tgt_id"]) for r in results]
    )

    if not all([n is not None for n in edge_datas]):
        logger.warning("Some edges are missing, maybe the storage is damaged")
    edge_degree = await asyncio.gather(
        *[knowledge_graph_inst.edge_degree(r["src_id"], r["tgt_id"]) for r in results]
    )
    edge_datas = [
        {"src_id": k["src_id"], "tgt_id": k["tgt_id"], "rank": d, **v}
        for k, v, d in zip(results, edge_datas, edge_degree)
        if v is not None
    ]
    edge_datas = sorted(
        edge_datas, key=lambda x: (x["rank"], x["weight"]), reverse=True
    )
    edge_datas = truncate_list_by_token_size(
        edge_datas,
        key=lambda x: x["description"],
        max_token_size=query_param.max_token_for_global_context,
    )

    use_entities = await _find_most_related_entities_from_relationships(
        edge_datas, query_param, knowledge_graph_inst
    )
    use_text_units = await _find_related_text_unit_from_relationships(
        edge_datas, query_param, text_chunks_db, knowledge_graph_inst
    )

    # 距離計算
    if use_text_units:
        cid_set = {u["id"] for u in use_text_units}
        dist_results = await chunks_vdb.query(keywords, top_k=len(cid_set) * 2, debug=False)
        dist_lookup = {r["id"]: r.get("distance") for r in dist_results if r["id"] in cid_set}
        for u in use_text_units:
            u["_distance"] = dist_lookup.get(u["id"], "N/A")

    # 取得したチャンクに対してメタデータフィルタを適用
    if query_param.metadata_filter:
        _before_cnt = len(use_text_units)
        _debug_log_filter_samples("Global", "before", use_text_units, query_param.metadata_filter)
        filtered_text_units = []
        for unit in use_text_units:
            chunk_metadata_raw = unit.get("metadata")
            chunk_metadata = {}
            if isinstance(chunk_metadata_raw, dict):
                chunk_metadata = chunk_metadata_raw
            elif isinstance(chunk_metadata_raw, str):
                try:
                    # 文字列の場合、JSONとしてパースを試みる
                    parsed_meta = json.loads(chunk_metadata_raw)
                    if isinstance(parsed_meta, dict):
                        chunk_metadata = parsed_meta
                except json.JSONDecodeError:
                    pass  # パース失敗時は空の辞書として扱う

            is_match = True
            for key, value in query_param.metadata_filter.items():
                if chunk_metadata.get(key) != value:
                    is_match = False
                    break
            if is_match:
                filtered_text_units.append(unit)
        use_text_units = filtered_text_units
        print(f"[Global] filter reduced candidates: before={_before_cnt} -> after={len(use_text_units)}")
        _debug_log_filter_samples("Global", "after", use_text_units, query_param.metadata_filter)

    # フィルタリングされたチャンクに基づいて、エンティティとリレーションシップの情報を再構築する
    filtered_chunk_content_map = {unit["id"]: unit["content"] for unit in use_text_units}

    # エンティティをフィルタリングして説明を再構築
    final_entities = []
    for entity in use_entities:
        source_ids = split_string_by_multi_markers(entity.get("source_id", ""), [GRAPH_FIELD_SEP])
        relevant_content = [
            filtered_chunk_content_map[sid] for sid in source_ids if sid in filtered_chunk_content_map
        ]
        if relevant_content:
            entity["description"] = GRAPH_FIELD_SEP.join(relevant_content)
            final_entities.append(entity)
    use_entities = final_entities

    # リレーションシップをフィルタリングして説明を再構築
    final_edge_datas = []
    for edge in edge_datas:
        source_ids = split_string_by_multi_markers(edge.get("source_id", ""), [GRAPH_FIELD_SEP])
        relevant_content = [
            filtered_chunk_content_map[sid] for sid in source_ids if sid in filtered_chunk_content_map
        ]
        if relevant_content:
            edge["description"] = GRAPH_FIELD_SEP.join(relevant_content)
            final_edge_datas.append(edge)
    edge_datas = final_edge_datas

    logger.info(
        f"Global query uses {len(use_entities)} entites, {len(edge_datas)} relations, {len(use_text_units)} text units"
    )
    relations_section_list = [
        ["id", "source", "target", "description", "keywords", "weight", "rank"]
    ]
    for i, e in enumerate(edge_datas):
        relations_section_list.append(
            [
                i,
                e["src_id"],
                e["tgt_id"],
                e["description"],
                e["keywords"],
                e["weight"],
                e["rank"],
            ]
        )
    relations_context = list_of_list_to_csv(relations_section_list)

    entites_section_list = [["id", "entity", "type", "description", "rank"]]
    for i, n in enumerate(use_entities):
        entites_section_list.append(
            [
                i,
                n["entity_name"],
                n.get("entity_type", "UNKNOWN"),
                n.get("description", "UNKNOWN"),
                n["rank"],
            ]
        )
    entities_context = list_of_list_to_csv(entites_section_list)

    text_units_section_list = [["id", "content"]]
    source = []  # sourceリストを追加
    for i, t in enumerate(use_text_units):
        text_units_section_list.append([i, t["content"]])
        source.append(t["content"])  # sourceリストに追加
    text_units_context = list_of_list_to_csv(text_units_section_list)

    context = f"""
-----Entities-----
```csv
{entities_context}
```
-----Relationships-----
```csv
{relations_context}
```
-----Sources-----
```csv
{text_units_context}
```
"""
    return context, source  # contextとsourceの両方を返す


async def _find_most_related_entities_from_relationships(
    edge_datas: list[dict],
    query_param: QueryParam,
    knowledge_graph_inst: BaseGraphStorage,
):
    entity_names = set()
    for e in edge_datas:
        entity_names.add(e["src_id"])
        entity_names.add(e["tgt_id"])

    node_datas = await asyncio.gather(
        *[knowledge_graph_inst.get_node(entity_name) for entity_name in entity_names]
    )

    node_degrees = await asyncio.gather(
        *[knowledge_graph_inst.node_degree(entity_name) for entity_name in entity_names]
    )
    node_datas = [
        {**n, "entity_name": k, "rank": d}
        for k, n, d in zip(entity_names, node_datas, node_degrees)
    ]

    node_datas = truncate_list_by_token_size(
        node_datas,
        key=lambda x: x["description"],
        max_token_size=query_param.max_token_for_local_context,
    )

    return node_datas


async def _find_related_text_unit_from_relationships(
    edge_datas: list[dict],
    query_param: QueryParam,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
):
    text_units = [
        split_string_by_multi_markers(dp["source_id"], [GRAPH_FIELD_SEP])
        for dp in edge_datas
    ]

    all_text_units_lookup = {}

    for index, unit_list in enumerate(text_units):
        for c_id in unit_list:
            if c_id not in all_text_units_lookup:
                all_text_units_lookup[c_id] = {
                    "data": await text_chunks_db.get_by_id(c_id),
                    "order": index,
                }

    if any([v is None for v in all_text_units_lookup.values()]):
        logger.warning("Text chunks are missing, maybe the storage is damaged")
    all_text_units = [
        {"id": k, **v} for k, v in all_text_units_lookup.items() if v is not None
    ]
    all_text_units = sorted(all_text_units, key=lambda x: x["order"])
    all_text_units = truncate_list_by_token_size(
        all_text_units,
        key=lambda x: x["data"]["content"],
        max_token_size=query_param.max_token_for_text_unit,
    )
    all_text_units: list[TextChunkSchema] = [t["data"] for t in all_text_units]

    # 時間フィルタを適用
    if query_param.start_time or query_param.end_time:
        filtered_by_time = []
        start_time_dt = datetime.fromisoformat(query_param.start_time) if query_param.start_time else None
        end_time_dt = datetime.fromisoformat(query_param.end_time) if query_param.end_time else None

        for unit in all_text_units:
            unit_time = unit.get("updated_at")
            if not unit_time:
                continue
            
            # JSONB から読み戻された場合、datetime は文字列になっている可能性がある
            if isinstance(unit_time, str):
                unit_time = datetime.fromisoformat(unit_time.replace('Z', '+00:00'))
            
            if start_time_dt and unit_time < start_time_dt:
                continue
            if end_time_dt and unit_time > end_time_dt:
                continue
            filtered_by_time.append(unit)
        all_text_units = filtered_by_time

    return all_text_units


async def hybrid_query(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    chunks_vdb: BaseVectorStorage,
    query_param: QueryParam,
    global_config: dict,
) -> str:
    low_level_context = None
    high_level_context = None
    use_model_func = global_config["llm_model_func"]

    kw_prompt_temp = PROMPTS["keywords_extraction"]
    kw_prompt = kw_prompt_temp.format(query=query)

    result = await use_model_func(kw_prompt)
    json_text = locate_json_string_body_from_string(result)
    try:
        keywords_data = json.loads(json_text)
        hl_keywords = keywords_data.get("high_level_keywords", [])
        ll_keywords = keywords_data.get("low_level_keywords", [])
        hl_keywords = ", ".join(hl_keywords)
        ll_keywords = ", ".join(ll_keywords)
    except json.JSONDecodeError:
        try:
            result = (
                result.replace(kw_prompt[:-1], "")
                .replace("user", "")
                .replace("model", "")
                .strip()
            )
            result = "{" + result.split("{")[1].split("}")[0] + "}"
            keywords_data = json.loads(result)
            hl_keywords = keywords_data.get("high_level_keywords", [])
            ll_keywords = keywords_data.get("low_level_keywords", [])
            hl_keywords = ", ".join(hl_keywords)
            ll_keywords = ", ".join(ll_keywords)
        # Handle parsing error
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            return PROMPTS["fail_response"], []
    if ll_keywords:
        low_level_context, ll_source = await _build_local_query_context(
            ll_keywords,
            knowledge_graph_inst,
            entities_vdb,
            chunks_vdb,
            text_chunks_db,
            query_param,
        )
        # None が返された場合の処理
        if low_level_context is None:
            ll_source = []
    else:
        low_level_context, ll_source = None, []

    if hl_keywords:
        high_level_context, hl_source = await _build_global_query_context(
            hl_keywords,
            knowledge_graph_inst,
            entities_vdb,
            relationships_vdb,
            chunks_vdb,
            text_chunks_db,
            query_param,
        )
        # None が返された場合の処理
        if high_level_context is None:
            hl_source = []
    else:
        high_level_context, hl_source = None, []

    context, _ = combine_contexts(high_level_context, low_level_context)
    # 重複を防ぐため、setを使用してユニークな要素のみを結合
    # 注意: これは文字列ベースの重複排除のため、完全に同じコンテンツのソースは1つにまとめられる
    # フィールド別チャンクと統合チャンクは異なるIDで保存されるが、同じコンテンツの場合、
    # 検索結果では文字列が同じため、この重複排除により1つにまとめられる
    source = list(set(ll_source + hl_source))

    if query_param.only_need_context:
        return context, source
    if context is None:
        return PROMPTS["fail_response"], []

    sys_prompt_temp = PROMPTS["rag_response"]
    sys_prompt = sys_prompt_temp.format(
        context_data=context, response_type=query_param.response_type
    )
    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
    )
    if len(response) > len(sys_prompt):
        response = (
            response.replace(sys_prompt, "")
            .replace("user", "")
            .replace("model", "")
            .replace(query, "")
            .replace("<system>", "")
            .replace("</system>", "")
            .strip()
        )
    return response, source


def combine_contexts(high_level_context, low_level_context):
    # Function to extract entities, relationships, and sources from context strings

    def extract_sections(context):
        entities_match = re.search(
            r"-----Entities-----\s*```csv\s*(.*?)\s*```", context, re.DOTALL
        )
        relationships_match = re.search(
            r"-----Relationships-----\s*```csv\s*(.*?)\s*```", context, re.DOTALL
        )
        sources_match = re.search(
            r"-----Sources-----\s*```csv\s*(.*?)\s*```", context, re.DOTALL
        )

        entities = entities_match.group(1) if entities_match else ""
        relationships = relationships_match.group(1) if relationships_match else ""
        sources = sources_match.group(1) if sources_match else ""

        return entities, relationships, sources

    # Extract sections from both contexts

    if high_level_context is None:
        warnings.warn(
            "High Level context is None. Return empty High entity/relationship/source"
        )
        hl_entities, hl_relationships, hl_sources = "", "", ""
    else:
        hl_entities, hl_relationships, hl_sources = extract_sections(high_level_context)

    if low_level_context is None:
        warnings.warn(
            "Low Level context is None. Return empty Low entity/relationship/source"
        )
        ll_entities, ll_relationships, ll_sources = "", "", ""
    else:
        ll_entities, ll_relationships, ll_sources = extract_sections(low_level_context)

    # Combine and deduplicate the entities

    combined_entities = process_combine_contexts(hl_entities, ll_entities)
    combined_entities = chunking_by_token_size(combined_entities, max_token_size=2000)
    # Combine and deduplicate the relationships
    combined_relationships = process_combine_contexts(
        hl_relationships, ll_relationships
    )
    combined_relationships = chunking_by_token_size(
        combined_relationships, max_token_size=2000
    )
    # Combine and deduplicate the sources
    combined_sources = process_combine_contexts(hl_sources, ll_sources)

    # Robustly extract content column, ignoring malformed lines
    source_lines = combined_sources.strip().split("\n")
    # Skip header row if present
    if source_lines and source_lines[0].lower().startswith("id"):
        source_lines = source_lines[1:]

    source = []
    for line in source_lines:
        line = line.strip()
        if not line:
            continue
        # Split only on first comma to keep remaining commas in content
        parts = line.split(",", 1)
        if len(parts) == 2:
            source.append(parts[1])
        # If format unexpected, skip or take entire line as fallback
        elif len(parts) == 1:
            source.append(parts[0])
    combined_sources = chunking_by_token_size(combined_sources, max_token_size=2000)
    # Format the combined context
    response_context = f"""
-----Entities-----
```csv
{combined_entities}
```
-----Relationships-----
```csv
{combined_relationships}
```
-----Sources-----
```csv
{combined_sources}
```
"""
    return response_context, source


async def naive_query(
    query,
    chunks_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
):
    use_model_func = global_config["llm_model_func"]
    results = await chunks_vdb.query(
        query,
        top_k=query_param.top_k,
        metadata_filter=query_param.metadata_filter,
        start_time=query_param.start_time,
        end_time=query_param.end_time,
    )
    if not len(results):
        return PROMPTS["fail_response"], []
    chunks_ids = [r["id"] for r in results]

    chunks = await text_chunks_db.get_by_ids(chunks_ids)

    maybe_trun_chunks = truncate_list_by_token_size(
        chunks,
        key=lambda x: x["content"],
        max_token_size=query_param.max_token_for_text_unit,
    )
    logger.info(f"Truncate {len(chunks)} to {len(maybe_trun_chunks)} chunks")
    section = "--New Chunk--\n".join([c["content"] for c in maybe_trun_chunks])
    source = [c["content"] for c in maybe_trun_chunks]
    if query_param.only_need_context:
        return section, source
    sys_prompt_temp = PROMPTS["naive_rag_response"]
    sys_prompt = sys_prompt_temp.format(
        content_data=section, response_type=query_param.response_type
    )
    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
    )

    if len(response) > len(sys_prompt):
        response = (
            response[len(sys_prompt) :]
            .replace(sys_prompt, "")
            .replace("user", "")
            .replace("model", "")
            .replace(query, "")
            .replace("<system>", "")
            .replace("</system>", "")
            .strip()
        )

    return response, source


async def path2chunk(
    scored_edged_reasoning_path, knowledge_graph_inst, pairs_append, query, max_chunks=5
):
    already_node = {}
    for k, v in scored_edged_reasoning_path.items():
        node_chunk_id = None

        for pathtuple, scorelist in v["Path"].items():
            if pathtuple in pairs_append:
                use_edge = pairs_append[pathtuple]
                edge_datas = []
                edge_datas = await asyncio.gather(
                    *[knowledge_graph_inst.get_edge(r[0], r[1]) for r in use_edge]
                )
                text_units = [
                    split_string_by_multi_markers(dp["source_id"], [GRAPH_FIELD_SEP])
                    for dp in edge_datas  # chunk ID
                ][0]

            else:
                use_edge = []
                text_units = []

            node_datas = await asyncio.gather(
                *[knowledge_graph_inst.get_node(pathtuple[0])]
            )
            for dp in node_datas:
                text_units_node = split_string_by_multi_markers(
                    dp["source_id"], [GRAPH_FIELD_SEP]
                )
                text_units = text_units + text_units_node

            node_datas = await asyncio.gather(
                *[knowledge_graph_inst.get_node(ents) for ents in pathtuple[1:]]
            )
            if query is not None:
                for dp in node_datas:
                    text_units_node = split_string_by_multi_markers(
                        dp["source_id"], [GRAPH_FIELD_SEP]
                    )
                    descriptionlist_node = split_string_by_multi_markers(
                        dp["description"], [GRAPH_FIELD_SEP]
                    )
                    if descriptionlist_node[0] not in already_node.keys():
                        already_node[descriptionlist_node[0]] = None

                        if len(text_units_node) == len(descriptionlist_node):
                            if len(text_units_node) > 5:
                                max_ids = int(max(5, len(text_units_node) / 2))
                                should_consider_idx = calculate_similarity(
                                    descriptionlist_node, query, k=max_ids
                                )
                                text_units_node = [
                                    text_units_node[i] for i in should_consider_idx
                                ]
                                already_node[descriptionlist_node[0]] = text_units_node
                    else:
                        text_units_node = already_node[descriptionlist_node[0]]
                    if text_units_node is not None:
                        text_units = text_units + text_units_node

            count_dict = Counter(text_units)
            total_score = scorelist[0] + scorelist[1] + 1
            for key, value in count_dict.items():
                count_dict[key] = value * total_score
            if node_chunk_id is None:
                node_chunk_id = count_dict
            else:
                node_chunk_id = node_chunk_id + count_dict
        v["Path"] = []
        if node_chunk_id is None:
            node_datas = await asyncio.gather(*[knowledge_graph_inst.get_node(k)])
            count_dict = Counter()
            for dp in node_datas:
                # ノードが存在しない場合はスキップ
                if dp is None:
                    continue
                text_units_node = split_string_by_multi_markers(
                    dp.get("source_id", ""), [GRAPH_FIELD_SEP]
                )
                count_dict.update(text_units_node)

            if count_dict:
                for _id, _ in count_dict.most_common(max_chunks):
                    v["Path"].append(_id)
            # v['Path'] = count_dict.most_common(max_chunks)#[]
        else:
            for id in node_chunk_id.most_common(max_chunks):
                v["Path"].append(id[0])
            # v['Path'] = node_chunk_id.most_common(max_chunks)
    return scored_edged_reasoning_path


def scorednode2chunk(input_dict, values_dict):
    for key, value_list in input_dict.items():
        input_dict[key] = [
            values_dict.get(val, None) for val in value_list if val in values_dict
        ]
        input_dict[key] = [val for val in input_dict[key] if val is not None]


def kwd2chunk(ent_from_query_dict, chunks_ids, chunk_nums):
    final_chunk = Counter()
    final_chunk_id = []
    for key, list_of_dicts in ent_from_query_dict.items():
        total_id_scores = Counter()
        id_scores_list = []
        id_scores = {}
        for d in list_of_dicts:
            if d == list_of_dicts[0]:
                score = d["Score"] * 2
            else:
                score = d["Score"]
            path = d["Path"]

            for id in path:
                if id == path[0] and id in chunks_ids:
                    score = score * 10
                if id in id_scores:
                    id_scores[id] += score
                else:
                    id_scores[id] = score
        id_scores_list.append(id_scores)

        for scores in id_scores_list:
            total_id_scores.update(scores)
        final_chunk = final_chunk + total_id_scores  # .most_common(3)

    for i in final_chunk.most_common(chunk_nums):
        final_chunk_id.append(i[0])
    return final_chunk_id


async def _build_mini_query_context(
    ent_from_query,
    type_keywords,
    originalquery,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    entity_name_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    chunks_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    embedder,
    query_param: QueryParam,
):
    imp_ents = []
    nodes_from_query_list = []
    ent_from_query_dict = {}

    for ent in ent_from_query:
        ent_from_query_dict[ent] = []
        results_node = await entity_name_vdb.query(
            ent,
            top_k=query_param.top_k,
            metadata_filter=query_param.metadata_filter,
            start_time=query_param.start_time,
            end_time=query_param.end_time,
        )
        # results_node の例（distance付き）
        # [{'entity_name': '"映画"', 'distance': 0.85}, {'entity_name': '"散歩"', 'distance': 0.72}, ...]

        nodes_from_query_list.append(results_node)
        ent_from_query_dict[ent] = [e["entity_name"] for e in results_node]

    candidate_reasoning_path = {}

    for results_node_list in nodes_from_query_list:
        candidate_reasoning_path_new = {
            key["entity_name"]: {"Score": key["distance"], "Path": []}
            for key in results_node_list
        }

        candidate_reasoning_path = {
            **candidate_reasoning_path,
            **candidate_reasoning_path_new,
        }
    for key in candidate_reasoning_path.keys():
        candidate_reasoning_path[key][
            "Path"
        ] = await knowledge_graph_inst.get_neighbors_within_k_hops(key, 2)
        imp_ents.append(key)

    short_path_entries = {
        name: entry
        for name, entry in candidate_reasoning_path.items()
        if len(entry["Path"]) < 1
    }
    sorted_short_path_entries = sorted(
        short_path_entries.items(), key=lambda x: x[1]["Score"], reverse=True
    )
    save_p = max(1, int(len(sorted_short_path_entries) * 0.2))
    top_short_path_entries = sorted_short_path_entries[:save_p]
    top_short_path_dict = {name: entry for name, entry in top_short_path_entries}
    long_path_entries = {
        name: entry
        for name, entry in candidate_reasoning_path.items()
        if len(entry["Path"]) >= 1
    }
    candidate_reasoning_path = {**long_path_entries, **top_short_path_dict}
    node_datas_from_type = await knowledge_graph_inst.get_node_from_types(
        type_keywords
    )  # entity_type, description,...

    maybe_answer_list = [n["entity_name"] for n in node_datas_from_type]
    imp_ents = imp_ents + maybe_answer_list
    scored_reasoning_path = cal_path_score_list(
        candidate_reasoning_path, maybe_answer_list
    )

    results_edge = await relationships_vdb.query(
        originalquery,
        top_k=len(ent_from_query) * query_param.top_k,
        metadata_filter=query_param.metadata_filter,
        start_time=query_param.start_time,
        end_time=query_param.end_time,
    )
    goodedge = []
    badedge = []
    for item in results_edge:
        if item["src_id"] in imp_ents or item["tgt_id"] in imp_ents:
            goodedge.append(item)
        else:
            badedge.append(item)
    scored_edged_reasoning_path, pairs_append = edge_vote_path(
        scored_reasoning_path, goodedge
    )
    scored_edged_reasoning_path = await path2chunk(
        scored_edged_reasoning_path,
        knowledge_graph_inst,
        pairs_append,
        originalquery,
        max_chunks=3,
    )

    entites_section_list = []
    node_datas = await asyncio.gather(
        *[
            knowledge_graph_inst.get_node(entity_name)
            for entity_name in scored_edged_reasoning_path.keys()
        ]
    )
    node_datas = [
        {**n, "entity_name": k, "Score": scored_edged_reasoning_path[k]["Score"]}
        for k, n in zip(scored_edged_reasoning_path.keys(), node_datas)
        if n is not None
    ]
    for i, n in enumerate(node_datas):
        entites_section_list.append(
            [
                n["entity_name"],
                n["Score"],
                n.get("description", "UNKNOWN"),
            ]
        )
    entites_section_list = sorted(
        entites_section_list, key=lambda x: x[1], reverse=True
    )
    entites_section_list = truncate_list_by_token_size(
        entites_section_list,
        key=lambda x: x[2],
        max_token_size=query_param.max_token_for_node_context,
    )

    entites_section_list.insert(0, ["entity", "score", "description"])
    entities_context = list_of_list_to_csv(entites_section_list)

    scorednode2chunk(ent_from_query_dict, scored_edged_reasoning_path)

    results = await chunks_vdb.query(
        originalquery,
        top_k=int(query_param.top_k / 2),
        metadata_filter=query_param.metadata_filter,
        start_time=query_param.start_time,
        end_time=query_param.end_time,
        debug=False,
    )
    chunks_ids = [r["id"] for r in results]
    final_chunk_id = kwd2chunk(
        ent_from_query_dict, chunks_ids, chunk_nums=int(query_param.top_k / 2)
    )

    if not len(results_node):
        return None, []

    if not len(results_edge):
        return None, []

    use_text_units = await asyncio.gather(
        *[text_chunks_db.get_by_id(id) for id in final_chunk_id]
    )
    text_units_section_list = [["id", "content"]]
    source = []

    for i, t in enumerate(use_text_units):
        if t is not None:
            text_units_section_list.append([i, t["content"]])
            source.append(t["content"])
    text_units_context = list_of_list_to_csv(text_units_section_list)

    response_context = f"""
-----Entities-----
```csv
{entities_context}
```
-----Sources-----
```csv
{text_units_context}
```
"""
    # When provenance is requested, return structured entities/chunks along with the context
    if query_param.include_provenance:
        entities_used = [
            {
                "entity_name": n.get("entity_name"),
                "score": n.get("Score"),
                "description": n.get("description", ""),
            }
            for n in node_datas
        ]

        chunks_used = []
        for cid, t in zip(final_chunk_id, use_text_units):
            if t is None:
                continue
            chunks_used.append(
                {
                    "chunk_id": cid,
                    "full_doc_id": t.get("full_doc_id"),
                    "chunk_order_index": t.get("chunk_order_index"),
                    "tokens": t.get("tokens"),
                    "content": t.get("content"),
                }
            )

        return {"context": response_context, "provenance": {"entities": entities_used, "chunks": chunks_used}}, source

    return response_context, source


async def minirag_query(  # MiniRAG
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    entity_name_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    chunks_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    embedder,
    query_param: QueryParam,
    global_config: dict,
) -> str:
    use_model_func = global_config["llm_model_func"]
    kw_prompt_temp = PROMPTS["minirag_query2kwd"]
    TYPE_POOL, TYPE_POOL_w_CASE = await knowledge_graph_inst.get_types()
    kw_prompt = kw_prompt_temp.format(query=query, TYPE_POOL=TYPE_POOL)
    result = await use_model_func(kw_prompt)
    # result の例2つ
    # ```json
    # {
    #   "answer_type_keywords": ["unknown"],
    #   "entities_from_query": ["映画"]
    # }
    # ```

    # ```json
    # {
    #   "answer_type_keywords": ["event", "person", "location"],
    #   "entities_from_query": ["映画"]
    # }
    # ```

    try:
        keywords_data = json_repair.loads(result)

        type_keywords = keywords_data.get("answer_type_keywords", [])
        entities_from_query = keywords_data.get("entities_from_query", [])[:5]

    except json.JSONDecodeError:
        try:
            result = (
                result.replace(kw_prompt[:-1], "")
                .replace("user", "")
                .replace("model", "")
                .strip()
            )
            result = "{" + result.split("{")[1].split("}")[0] + "}"
            keywords_data = json_repair.loads(result)
            type_keywords = keywords_data.get("answer_type_keywords", [])
            entities_from_query = keywords_data.get("entities_from_query", [])[:5]

        # Handle parsing error
        except Exception as e:
            print(f"JSON parsing error: {e}")
            return PROMPTS["fail_response"], []

    context, source = await _build_mini_query_context(
        entities_from_query,
        type_keywords,
        query,
        knowledge_graph_inst,
        entities_vdb,
        entity_name_vdb,
        relationships_vdb,
        chunks_vdb,
        text_chunks_db,
        embedder,
        query_param,
    )

    # Unpack provenance-aware context if requested
    provenance = None
    context_str = context
    if isinstance(context, dict) and "context" in context:
        provenance = context.get("provenance")
        context_str = context.get("context")

    if query_param.only_need_context:
        if query_param.include_provenance:
            return {"context": context_str, "provenance": provenance}, source
        return context_str, source
    if context_str is None:
        return PROMPTS["fail_response"], []

    sys_prompt_temp = PROMPTS["rag_response"]
    sys_prompt = sys_prompt_temp.format(
        context_data=context_str, response_type=query_param.response_type
    )
    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
    )
    if query_param.include_provenance:
        return {"answer": response, "provenance": provenance}, source
    return response, source