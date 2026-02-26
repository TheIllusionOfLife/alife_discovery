# Objective-free Lives with Assembly Theory (Assembly Index) — ALIFE Research Plan

> **Goal (original vision):** Run many simulations **without objective functions**, apply only **minimal filters**, and **discover life-like entities**. “Life-like” is defined **only by Assembly Theory** (assembly index + copy number / assembly) — no extra biology-inspired criteria for now.

This document proposes a concrete research program and paper plan for an **ALIFE conference submission**, designed to produce a **step-change** improvement (not incremental) by making *entity discovery* the centerpiece and treating “agent observation capability” as *just one accelerator lever* among several.

---

## 1) Core research claim (what we want the paper to say)

### 1.1 Primary claim

**Objective-free discovery produces life-like entities** (high assembly index with nontrivial copy number) **at measurable rates**, and these rates exhibit **phase-diagram structure** across accelerator levers.

### 1.2 Secondary claim

“Observation capability” is **one** accelerator lever, but not the thesis. The thesis is:

* **Discovery protocol** (randomized rules + minimal filters + observatory)
* **Assembly Theory measurement** (assembly index + copy number / assembly)
* **Emergent entity ecology** (distribution, recurrence, and assembly growth)

---

## 2) Assembly Theory operationalization

### 2.1 What counts as an “object”

Assembly Theory requires definable objects that can be decomposed and (in principle) constructed from building blocks.

We will treat each **entity** as an **object graph**:

* Nodes = building blocks (typed)
* Edges = bonds/joins between blocks

Entities exist **in space**, but measurement is **over their object graph**.

### 2.2 What we measure

We will measure, per unique object type *i*:

* **Assembly index** `a_i`: minimal number of joins to assemble the object from building blocks, allowing reuse of repeated substructures.
* **Copy number** `n_i`: number of occurrences of that object type observed in a run (or in a time window).

Optionally, we also compute **Assembly (capital A)** for the ensemble at a given time window: a population-level summary that emphasizes **objects that are both complex and repeatedly realized**.

### 2.3 “Life-like” definition for this project

For the first ALIFE version, we commit to:

* **Life-like = Assembly Theory only**.

Operationally, an entity is “life-like” if it appears in a run and satisfies a thresholding rule based on `a_i` and `n_i`.

**Important:** We should avoid absolute universal thresholds (“`a_i > 15` is life”) and instead report:

* Distributions of `(a_i, n_i)`
* Tail behavior (max `a_i` at given `n_i`)
* Recurrence across seeds / runs

---

## 3) Simulation redesign (to make assembly index natural *and* visually life-like)

Goal: **building blocks should visually look life-like when assembly index is meaningful**.

### 3.1 Recommended building blocks: “cellular tiles with membranes”

Use blocks that naturally evoke living structure:

* **Membrane tile**: edge-like block that can form closed loops
* **Cytosol tile**: interior filler block
* **Catalyst tile**: special block that enables or biases join operations
* **Nutrient tile** (optional): transient block that can be incorporated/consumed

These blocks allow entities to look like:

* vesicles / protocells
* segmented worms
* colonies
* budding structures

### 3.2 Spatial embedding

Use a 2D lattice (torus is fine) with **local interaction rules**, but the state is not just a scalar. Each spatial site can hold:

* either empty
* or a pointer to an object (entity id) + block type at that location

Entities therefore have both:

* a **graph representation** (for assembly index)
* a **spatial footprint** (for visual inspection + contact dynamics)

### 3.3 Physics-lite join operations (AT-friendly)

Define explicit join operations between two objects:

* **Join**: when two blocks (or boundary sites) are adjacent, they can bond with a rule-dependent probability.
* **Split** (optional): bonds can break under noise or stress.
* **Move**: entities drift via local moves, enabling collisions.

Crucially, the world’s fundamental operation is an AT-style **join**.

### 3.4 Minimal filters (objective-free)

We keep filters minimal and symmetric (not optimizing assembly):

* **Nontrivial persistence**: the world does not collapse to empty immediately.
* **Nontrivial diversity**: not all blocks become a single trivial type.
* **No runaway**: bound maximum entity size or energy to avoid uninformative infinite growth.

These are *viability constraints*, not positive objectives.

---

## 4) Rule space (how we randomize without objectives)

### 4.1 Shared rule tables (simple & scalable)

Each simulation run samples a random rule set governing:

* join probabilities conditioned on local block types
* movement biases
* bond breaking under local conditions
* catalyst effects

Keep the rule encoding small enough that we can sample millions.

### 4.2 Where “observation capability” fits

If you want to preserve your existing framing, reinterpret “observation capability” as:

* **Information available to local update rules**

But it becomes **just one lever** among others (update order, noise, resources, catalysts).

---

## 5) Entity observatory (automatic detection + AT metrics)

### 5.1 Entity segmentation

At each timestep (or window), detect entities by:

* connected components over bonded blocks
* assign canonical representation (graph isomorphism / normalized form)

### 5.2 Tracking over time

Track entity lineages by overlap / matching of subgraphs:

* parent–child link if a split event occurs or if a large overlap persists

This lineage tracking is optional for ALIFE v1, but extremely valuable.

### 5.3 Assembly index computation

We will compute `a_i` using:

* **Exact** if feasible for small objects
* **Approximate upper bound** for larger objects (branch-and-bound, greedy reuse, or split-branch heuristic)

We must report:

* approximation method
* validation on small objects where exact is computable

### 5.4 Copy number and recurrence

Compute `n_i` per run and across runs:

* per-run copy number
* cross-run recurrence count
* stability across seeds

---

## 6) Main experiments (designed to create a “drastic” leap)

### Experiment 1 — Discovery baseline

**Question:** Does objective-free sampling produce high-assembly objects with nontrivial copy number?

* Run N rule samples × S seeds
* Collect `(a_i, n_i)` cloud
* Report maxima, tail exponents, and recurrence

**Key figure:** scatter or heatmap of `(a_i, log n_i)` with density contours.

---

### Experiment 2 — Accelerator lever phase diagram

**Question:** Which levers accelerate life-like discovery, and how?

Candidate levers (choose 2–3 for the paper):

1. Observation capability (local info)
2. Update schedule (synchronous ↔ asynchronous mix)
3. Environmental openness (resource field on/off)
4. Noise (bond break / action noise)
5. Catalysts (present/absent)

For each lever setting:

* measure distribution shift in `(a_i, n_i)`
* measure probability of discovering objects above a high-quantile frontier

**Key figure:** a 2D phase diagram where color = probability of discovering objects with `a_i >= A*` and `n_i >= N*`.

---

### Experiment 3 — Mechanistic sanity checks (to defend against “artifact” critique)

1. **Shuffle / randomization controls**

   * randomize bonds while keeping block counts → assembly index should drop or patterns should change.
2. **Alternative object definitions**

   * test robustness of canonicalization (rotation/translation invariance)
3. **Computational bias audit**

   * show that approximations in `a_i` do not reorder the top-k objects drastically.

---

## 7) Paper structure for ALIFE

### Title candidates

* **“Objective-free Lives: Discovering High-Assembly Entities via Minimal Filters”**
* “Assembly Theory as an Observatory for Objective-free Artificial Life”

### Abstract (template)

* Problem: objective-driven fitness can bias what we find
* Method: objective-free sampling + minimal filters + assembly observatory
* Result: high-assembly, repeated entities appear; discovery shows phase diagram across levers
* Contribution: a scalable protocol for life-like discovery without objectives

### Results section (order)

1. Discovery baseline: high-assembly repeated entities exist
2. Phase diagram: which levers accelerate discovery
3. Example gallery: representative high-assembly entities (visuals)
4. Controls/robustness: not trivial artifacts

### Figures (must-have)

1. Entity gallery (most persuasive)
2. `(a_i, n_i)` distribution plot
3. Phase diagram for discovery probability
4. Methods schematic (pipeline)

---

## 8) Engineering plan (minimum viable implementation)

### 8.1 Two-track implementation

**Track A (fast):** keep your current grid simulator and add an “object graph” layer:

* treat connected spatial clusters as graphs
* compute approximate assembly index

**Track B (clean AT):** implement explicit join/split physics as the native dynamics

Recommendation: prototype in Track A in 1–2 weeks, then migrate to Track B for the paper.

### 8.2 Compute budgeting

* Many short runs > few long runs
* Early stopping if the world becomes trivial
* Incremental observatory sampling (only snapshot every K steps)

---

## 9) Risks and how we mitigate them

1. **Assembly index too expensive**

   * Mitigation: cap entity size; approximate; exact on small objects; batch compute

2. **High assembly but visually not life-like**

   * Mitigation: choose membrane-like blocks; include catalysts; include resource field; curated gallery

3. **High copy number but low novelty**

   * Mitigation: track unique object types; report diversity (unique count at high `a_i`)

4. **Reviewer skepticism: “still selection”**

   * Mitigation: explicitly frame as minimal-criteria viability; show phase diagrams and robustness; emphasize no positive objective

---

## 10) Questions for you (so I can update this plan)

### Q1 — Block vocabulary

Which block palette do you prefer?

* (A) Membrane + cytosol + catalyst (minimal)
* (B) Add nutrient + waste (metabolism-like)
* (C) Add polarity/charge to membrane blocks (directional assembly)

### Q2 — What is the join primitive?

Choose one as the “chemistry” of the world:

* (A) Adjacent blocks bond with probability from rule table
* (B) Bond requires catalyst nearby
* (C) Bond requires resource locally available

### Q3 — Minimal filter exact definition

Which minimal filter is acceptable?

* (A) Survive T steps + nontrivial block diversity
* (B) Above plus at least one entity reaches size >= K
* (C) Above plus at least M entities exist at end

### Q4 — What counts as “copy number”?

* (A) Same object appears simultaneously at a timestep
* (B) Same object appears at any time within a run (time-integrated)
* (C) Same object recurs across different runs (cross-run recurrence)

### Q5 — Accelerator levers for the paper

Pick 2–3 levers you most want to highlight:

* Observation capability
* Update schedule (async vs sync)
* Resources / openness
* Catalysts
* Noise

---

## 11) Large-Scale Experiment Results (PR #10)

### 11.1 Experimental parameters

| Parameter | Small-scale (PR #9) | Large-scale (PR #10) |
|-----------|---------------------|----------------------|
| Rule samples | 100 | 1,000 |
| Seeds per rule | 3 | 5 |
| Steps per sim | 200 | 500 |
| Null shuffles | 20 | 20 |
| Total sim-steps | 6,000,000 | 250,000,000 |

### 11.2 Key findings

#### Experiment 1 — Discovery baseline (1000 rules × 5 seeds × 500 steps)

- **7,079,166 entity observations** (41× the small-scale run)
- Assembly index: min=0, max=6, mean=0.06 (unchanged from small-scale)
- Copy number: min=1, max=15, mean=9.83 (unchanged)
- Entity size distribution: 94.6% size=1, 4.9% size=2, 0.4% size=3, 0.04% size=4, <0.01% size=5–6
- Max entity size observed: 6 (only 18 instances out of 7M observations)

#### Entity gallery (1000 rules × 5 seeds × 500 steps)

- **282 unique entity types** (up from 72 at 100 rules — 3.9× increase)
- Top-10 entity ranking is **stable across scales** (same hash values, same order)
- Top entities: all size 2 (a_i=1) with very high copy counts (up to 108,352)
- Rank 7–10: size 3 entities with a_i=2
- **No entities with a_i ≥ 3 in top-10**
- Copy counts scale linearly with compute (≈40× increase, matching ≈41.7× more sim-steps)

#### Assembly audit (1000 rules × 5 seeds × 500 steps × 20 null shuffles)

- **0.0% significant excess** (a_i > null_mean + 2σ): assembly is **entirely size-driven**
- Observed mean a_i = 0.0595; null mean a_i = 0.0595; enrichment = 0.0000
- Zero excess at every entity size (1–6)
- Result is robust: confirmed at both 170K and 7M observation scales

### 11.3 Comparison: small-scale vs large-scale

| Metric | 100 rules | 1,000 rules | Change |
|--------|-----------|-------------|--------|
| Total observations | 170,192 | 7,079,166 | 41× |
| Unique entity types | 72 | 282 | 3.9× |
| Max entity size | 6 | 6 | unchanged |
| Max a_i | 6 | 6 | unchanged |
| Mean a_i | 0.0576 | 0.0595 | +3.3% |
| Significant excess | 0.0% | 0.0% | unchanged |
| Top entity a_i | 1 | 1 | unchanged |
| Top entity copy count | 2,767 | 108,352 | 39× |

### 11.4 Interpretation and paper narrative

The large-scale experiments provide a **robust negative result**: 10× more rule diversity and 2.5× longer simulations produce no structurally non-trivial assembly. All observed assembly index is fully explained by entity size (the null model matches observations exactly at every scale).

This is itself a **meaningful contribution** to the ALIFE literature:

1. **Objective-free rule sampling produces entities, but not complex ones.** The system reliably generates entities (282 types at scale), but the entity ecology is dominated by size-1 and size-2 objects. The tail of the size distribution falls off steeply.

2. **Assembly index is size-driven in this regime.** The shuffle-bond null model shows zero excess assembly at every entity size, meaning the observed a_i values are entirely explained by the number of edges in the entity graph, not by any structural specificity.

3. **Copy number scales linearly with compute but does not create complexity.** High copy numbers (>100K) arise from the most common dimers, not from structurally interesting entities. More computation produces more copies, not more complex entities.

4. **The entity type ranking is robust.** The same entity types dominate at both 100 and 1000 rules, suggesting the system's emergent ecology is deterministic and convergent.

#### Revised paper claim

The original claim — “objective-free rule sampling produces life-like entities at measurable rates” — should be revised to:

> **Objective-free rule sampling in a block world with local bonding rules produces a stable entity ecology with measurable assembly indices, but the observed assembly is entirely size-driven. Structurally non-trivial assembly (a_i exceeding null expectations) does not emerge under uniform random rule sampling, even at large scale (10³ rules × 5 seeds × 500 steps).**

This reframes the paper as a **characterization of the boundary conditions for emergent complexity**: the system shows where assembly *doesn't* spontaneously emerge, which constrains future search strategies (e.g., biased rule sampling, larger grids, catalytic mechanisms, or environmental gradients).

### 11.5 Output artifacts

| Artifact | Path |
|----------|------|
| Experiment 1 scatter | `data/experiment1_large/figures/experiment1_ai_cn_scatter.pdf` |
| Experiment 1 heatmap | `data/experiment1_large/figures/experiment1_ai_cn_heatmap.pdf` |
| Experiment 1 size dist | `data/experiment1_large/figures/experiment1_size_dist.pdf` |
| Entity gallery | `data/entity_gallery_large/entity_gallery.pdf` |
| Gallery metadata | `data/entity_gallery_large/entity_gallery_meta.csv` |
| Audit dist figure | `data/assembly_audit_large/figures/assembly_audit_dist.pdf` |
| Audit excess figure | `data/assembly_audit_large/figures/assembly_audit_excess.pdf` |
| Audit summary | `data/assembly_audit_large/audit_summary.txt` |
| Exp 1 combined data | `data/experiment1_large/entity_log_combined.parquet` |
| Audit combined data | `data/assembly_audit_large/entity_log_combined.parquet` |
