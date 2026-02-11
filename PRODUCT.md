# PRODUCT.md

## Purpose

`objectless_alife` is a research-oriented product that explores whether structured, non-trivial dynamics can emerge in multi-agent systems without explicit objective functions.

## Target Users

- ALife and complex systems researchers
- Engineers prototyping open-ended simulation pipelines
- Technical collaborators analyzing phase-based observation designs

## Core Value Proposition

- Objective-free exploration with reproducible simulation behavior
- Side-by-side comparison between two observation models
- Analysis-ready output artifacts for downstream interpretation

## Key Features

- Seeded rule generation and deterministic simulation replayability
- Two-phase observation design:
  - Phase 1: local density only
  - Phase 2: density + dominant neighbor state profile
- Physical inconsistency filtering (halt and state-uniform termination)
- Optional dynamic filters for ablation experiments
- Rich metric extraction and Parquet-based logging
- Animation rendering for qualitative behavior review

## Business/Research Objectives

- Validate feasibility of objective-free search under constrained world dynamics
- Quantify differences between phase designs using shared experimental protocol
- Produce reproducible artifacts suitable for further statistical analysis and publication decisions

## Non-Goals

- No reward optimization or fitness-based selection in the core loop
- No production service/API layer at this stage
- No interactive UI requirement for primary workflow
