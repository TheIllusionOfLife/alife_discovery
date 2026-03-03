"""Parallel rule-level simulation runner using multiprocessing.

Each rule seed is fully independent, so we can run them in parallel across
CPU cores. Each worker process gets its own ``_ASSEMBLY_CACHE`` (module-level
dict in ``metrics.assembly``) since multiprocessing forks fresh imports.
"""

from __future__ import annotations

import os
import random
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from alife_discovery.config.types import BlockWorldConfig

_WorkerResult = tuple[list[dict[str, Any]], dict[str, Any]]


def _run_single_rule(args: tuple[BlockWorldConfig, int, int]) -> _WorkerResult:
    """Worker function: run one rule seed and return (entity_records, summary).

    Imports are inside the function so each forked process gets a fresh
    module-level ``_ASSEMBLY_CACHE``.
    """
    cfg, rule_seed, sim_seed = args

    from alife_discovery.config.constants import ENTITY_SNAPSHOT_INTERVAL
    from alife_discovery.domain.block_world import (
        BlockRuleTable,
        BlockWorld,
        PartnerRuleTable,
        generate_block_rule_table,
        generate_partner_specific_rule_table,
    )
    from alife_discovery.domain.entity import detect_entities
    from alife_discovery.metrics.assembly import compute_entity_metrics

    run_id = f"bw_rs{rule_seed}_ss{sim_seed}"
    rule_table: BlockRuleTable | PartnerRuleTable
    if cfg.partner_specific_rules:
        rule_table = generate_partner_specific_rule_table(rule_seed)
    else:
        rule_table = generate_block_rule_table(rule_seed)
    rng = random.Random(sim_seed)
    world = BlockWorld.create(cfg, rng)

    entity_records: list[dict[str, Any]] = []

    for step in range(cfg.steps):
        world.step(rule_table, cfg.noise_level, rng, update_mode=cfg.update_mode)

        if (step + 1) % ENTITY_SNAPSHOT_INTERVAL == 0 or step == cfg.steps - 1:
            entities = detect_entities(world)
            records = compute_entity_metrics(
                entities,
                step=step,
                run_id=run_id,
                n_null_shuffles=cfg.n_null_shuffles,
                compute_reuse=cfg.compute_reuse_index,
            )
            entity_records.extend(records)

    final_entities = detect_entities(world)
    summary = {
        "run_id": run_id,
        "rule_seed": rule_seed,
        "sim_seed": sim_seed,
        "n_entities_final": len(final_entities),
        "n_bonds_final": len(world.bonds),
    }

    return entity_records, summary


def run_rules_parallel(
    n_rules: int,
    out_dir: Path,
    config: BlockWorldConfig | None = None,
    n_workers: int | None = None,
) -> list[dict[str, Any]]:
    """Run block-world simulations in parallel across rule seeds.

    Drop-in replacement for ``run_block_world_search`` with identical output
    format (entity_log.parquet + run summaries).

    Args:
        n_rules: Number of rule seeds to evaluate.
        out_dir: Output directory for logs.
        config: Block world configuration.
        n_workers: Number of parallel workers. Defaults to ``os.cpu_count() - 1``.

    Returns:
        List of run summary dicts (same as ``run_block_world_search``).
    """
    if n_rules < 1:
        raise ValueError("n_rules must be >= 1")

    cfg = config or BlockWorldConfig()
    if n_workers is None:
        cpu = os.cpu_count() or 2
        n_workers = max(1, cpu - 1)

    logs_dir = Path(out_dir) / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Build worker arguments
    worker_args = [(cfg, cfg.rule_seed + i, cfg.sim_seed + i) for i in range(n_rules)]

    # Run in parallel
    all_entity_records: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    with Pool(processes=n_workers) as pool:
        for entity_records, summary in pool.map(_run_single_rule, worker_args):
            all_entity_records.extend(entity_records)
            summaries.append(summary)

    # Write entity log with correct schema
    if all_entity_records:
        from alife_discovery.io.schemas import (
            ENTITY_LOG_SCHEMA,
            ENTITY_LOG_SCHEMA_FULL,
            ENTITY_LOG_SCHEMA_WITH_NULL,
            ENTITY_LOG_SCHEMA_WITH_REUSE,
        )

        has_null = cfg.n_null_shuffles > 0
        has_reuse = cfg.compute_reuse_index
        if has_null and has_reuse:
            schema = ENTITY_LOG_SCHEMA_FULL
        elif has_null:
            schema = ENTITY_LOG_SCHEMA_WITH_NULL
        elif has_reuse:
            schema = ENTITY_LOG_SCHEMA_WITH_REUSE
        else:
            schema = ENTITY_LOG_SCHEMA

        entity_log_path = logs_dir / "entity_log.parquet"
        table = pa.Table.from_pylist(all_entity_records, schema=schema)
        with pq.ParquetWriter(entity_log_path, schema) as writer:
            writer.write_table(table)

    return summaries
