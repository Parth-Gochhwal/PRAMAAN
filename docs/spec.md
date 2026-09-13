# PRAMAAN

### Sovereign, Structure-Aware, Precision-Adaptive, Proof-Carrying Optimization Runtime

**SIH Problem Statement ID26119 (MRPL) — Indigenous GPU-Accelerated LP/MILP/QP Solver**

> **Tagline:** *Compute Fast. Polish Precisely. Prove Independently.*

---

## 0. Purpose of this document

This is the **single canonical source of truth** for the project's solution architecture. Six team members will each be using multiple AI assistants (Claude, ChatGPT, Gemini) to help draft slides, code, and documentation. Without a shared reference, every assistant will "helpfully" reinvent parts of the architecture, drift toward the generic consensus solution everyone else is submitting, or introduce naming/scope inconsistencies across slides and code.

**Before asking any AI assistant to help with any part of this project — slides, code, benchmarks, reports — paste or attach this document as context first.** Treat any suggestion that contradicts this document (different core algorithm choice, different USP framing, different component names) as something to flag to the team before adopting, not something to silently merge in.

---

## 1. Why this architecture, and not the "obvious" one

Every team competing for this PS is going to feed the problem statement into Claude/ChatGPT/Gemini. All three models, independently, converge on the same answer: *presolve → sparse CSR storage → CPU simplex + GPU PDHG for LP → CPU branch-and-bound with GPU-batched node relaxations for MILP → benchmark against MIPLIB/Netlib/Mittelmann → refinery blending demo → "sovereignty" framed as source-code ownership.* That convergence was confirmed explicitly by cross-checking all three models against each other's output.

That architecture is **not wrong** — it's the correct technical foundation. But it is also what most of the room will present, because it's the modal answer any LLM gives to this prompt. Two problems follow from adopting it as your headline pitch:

1. **You'd be racing on the one metric where you cannot win.** A from-scratch GPU LP solver competing on cold-start MIPLIB benchmark speed is competing against 30 years of Gurobi/CPLEX engineering and a published OR paper (cuPDLP.jl) that took real research effort to barely match commercial solvers. A hackathon team will not out-benchmark that, and a technically literate judge will know it.
2. **It answers "how do we solve this fast?" but never "why should the refinery trust the answer?"** — which is the actual question a real refinery, and a real judge, will ask about any solver making decisions that feed into costing, custody transfer, and safety-adjacent scheduling.

PRAMAAN keeps the consensus architecture as its **foundation** (it has to — the PS explicitly requires the numerical core), but makes the **headline differentiator** something none of the surveyed AI outputs made central on their own: **structure discovery, precision-adaptive execution, and independently verifiable proof of correctness** — with GPU parallelism earned honestly through decomposition and scenario-batching rather than brute-forced onto operations (simplex, B&B) that are fundamentally sequential/irregular.

---

## 2. The four pillars

### PILLAR 1 — STRUCTURE

*Exploit the mathematics hidden inside the industrial model instead of treating it as an undifferentiated sparse matrix.*

Refinery and supply-chain optimization models are not arbitrary sparse matrices — they have repeated structure across time periods, units, products, and tanks (block-angular / staircase structure: independent per-period or per-unit blocks coupled by a small set of linking constraints, like inventory carryover). A **Structural Fingerprint Engine** analyzes the parsed model as a bipartite graph (variable nodes, constraint nodes, edges = non-zero coefficients) and detects:

- Temporal blocks / staircase structure
- Near-independent components
- Weakly-coupled blocks (candidates for decomposition)
- Coefficient range / numerical risk

Output is a **model fingerprint report** (variables, constraints, sparsity, block-structure score, numerical risk, recommended solve strategy) — genuinely demo-friendly and immediately shows judges the system "understands" the problem rather than blindly grinding on it.

If strong block structure is detected, the model is split via **Benders decomposition** (chosen over Dantzig-Wolfe/Lagrangian for scope discipline — Benders fits naturally when refinery unit/period subproblems are coupled through a small set of linking constraints, and is simpler to implement correctly in hackathon time): a CPU master problem handles coupling/branching decisions, while independent unit/period subproblems are solved as **parallel GPU-friendly blocks**. This is the honest version of "GPU-accelerated MILP" — instead of trying to parallelize one irregular branch-and-bound tree (a genuinely unsolved research problem), you create genuinely independent parallel work by decomposing the model first.

### PILLAR 2 — SPEED

*Heterogeneous CPU/GPU execution, spending precision and re-computation only where actually necessary.*

The PS explicitly names both **revised simplex and interior-point methods** as core continuous-optimization algorithms. PRAMAAN implements both from scratch on CPU: dual/revised simplex as the default LP path, and a primal-dual interior-point/barrier method as the alternative for large well-conditioned LPs and as the QP engine's core method (interior-point extends naturally to the quadratic objective; this also directly covers the QP requirement without inventing a separate method). On top of this required simplex+IPM core sit two lightweight control-flow layers — not new algorithms in themselves, hence low engineering risk relative to their differentiation value:

**(a) Precision Ladder.** The team's actual hardware (consumer RTX-class GPUs, not HPC accelerators) is not built primarily for FP64 throughput. Rather than assume every GPU computation needs double precision, PRAMAAN runs a fast **FP32 GPU pass** (sparse PDHG) first, checks the residual, and only escalates to a **CPU FP64 polish** when the fast pass isn't accurate enough. Applied inside branch-and-bound: a node only needs enough precision to decide whether it can be pruned — a rough bound is often sufficient, and only promising nodes get high-precision refinement. This is both a genuine hardware-appropriate engineering decision (directly justified by the team's actual GPUs) and a real compute saving.

**(b) Warm-Start / Incremental Resolve Engine.** MRPL does not solve one LP once — it re-solves the same structural scheduling problem repeatedly (daily/weekly) as crude assay, prices, and demand forecasts update. Every standard benchmark (Netlib, MIPLIB, Mittelmann) measures **cold-start** solve time on a single static instance, which is the wrong metric for how this system is actually operated. PRAMAAN caches the optimal basis and detects the "delta" between successive problem instances (same structure, changed RHS/bounds/costs), then hot-starts dual simplex from the cached basis instead of solving cold each time. This produces a benchmark story nobody else will have: **cumulative time-to-solution over a simulated rolling 30-day rescheduling trace**, comparing cold-restart-every-day (what a naive/generic solver does) against PRAMAAN's warm-start — a realistic 5–10x cumulative advantage that maps directly to MRPL's actual operating pattern.

### PILLAR 3 — ROBUSTNESS UNDER UNCERTAINTY *(P2 bonus layer — see scope note)*

*Use the GPU for what it's actually good at: many independent solves, not one sequential one.*

**Scope note (read before pitching this section):** the PS asks for a deterministic LP/MILP/QP solver benchmarked on MIPLIB/Netlib/QPLIB — it does not ask for stochastic/robust optimization. This pillar is explicitly **P2, built only after P0/P1 are complete and demoable**, and it reuses the exact same solve engine and warm-start machinery with no new algorithm underneath — so it costs no separate engineering track. Pitch it as *"what this architecture additionally enables once the required engine works,"* never as a substitute for, or equal in weight to, the core deliverable. If time is short, this is the first thing to cut from the live demo without weakening the submission's compliance with the PS.

Simplex is inherently sequential; branch-and-bound trees are irregular and hard to load-balance on GPUs — this is real, acknowledged research difficulty, not a solved problem, and claiming otherwise is a credibility risk in front of informed judges. Pillar 1's decomposition creates one source of legitimate GPU parallelism (independent blocks of a single model). Pillar 3 creates a second, complementary source: refinery inputs are genuinely uncertain (crude assay quality varies by cargo, product prices fluctuate, demand forecasts carry error). PRAMAAN's **GPU Scenario-Parallel Engine** Monte Carlo-samples perturbations of the input data and solves **hundreds to thousands of perturbed instances simultaneously** on GPU (each scenario as independent, embarrassingly-parallel work, warm-started from the nominal solution's basis). The output is not one brittle deterministic plan but a **distribution over refinery margin / a robustness view** — a hedged production plan under real uncertainty, which is a genuine capability gap versus AspenTech/CPLEX-based tools that hand MRPL a single deterministic answer per run.

### PILLAR 4 — TRUST

*Make every industrial decision independently verifiable — not just fast.*

This is the sharpest differentiator, and the one no surveyed AI output made central. For a refinery, the real risk is not "the solve took 30 seconds too long" — it's "the solver reports optimal, but a numerical error or implementation bug produced an infeasible schedule that gets acted on." The PS explicitly calls out numerical robustness, degeneracy, and ill-conditioning as core requirements; PRAMAAN answers this architecturally rather than with a vague "we monitor residuals" claim:

- **The solver and the verifier are separate programs.** Asking the same code "are you correct?" proves nothing. `pramaan-solve` emits a solution **plus a certificate**; `pramaan-verify` is a small, independently-auditable program that re-checks it from scratch.
- **Certificate contents (P0 scope, a practical subset of the VIPR certificate concept, not the full academic standard):** primal residual (`Ax − b`), dual residual, complementarity residual, integrality check (MILP), reconstructed pre-presolve solution, objective bound gap.
- **Reversible Transformation Ledger:** every presolve step (fixed-variable elimination, singleton substitution, row scaling, bound tightening) is logged as a replayable transformation, not silently applied and forgotten — this is what makes reconstructing and verifying the original-space solution possible at all.
- **Live demo value:** solve an instance, show `CERTIFICATE VALID`, then deliberately tamper with the reported solution and show the independent verifier immediately reject it (`CERTIFICATE INVALID`). This is a concrete, memorable, judge-legible proof that the "trust" claim isn't just a slide bullet.

Exact/rational-arithmetic verification (rather than floating-point residual checks) is a credible **stretch goal (P2)** — checking optimality of a *given* basis is far cheaper than solving from scratch, so it's feasible, but it is not required for a convincing P0 certificate.

---

## 3. Sovereignty, defined precisely (not just "we wrote the code")

Most competing teams will define sovereignty as "the code is Indian and open source." PRAMAAN uses three explicit levels, because the PS's "not built upon any existing solver library" clause demands more than that framing:

- **Level 1 — Source sovereignty:** we own the solver codebase.
- **Level 2 — Algorithmic sovereignty:** we own and understand the optimization algorithms themselves rather than wrapping HiGHS/SCIP/CBC/OR-Tools/Gurobi/CPLEX internals — this is what the PS explicitly requires, and any dependency on those libraries, even indirect, is disqualifying.
- **Level 3 — Decision sovereignty:** MRPL can independently establish *why* a given result should be trusted, via the certificate/verifier — not just inspect the source, but verify each specific decision.

**Hardware sovereignty note:** CUDA is NVIDIA-controlled, and the team's GPUs are NVIDIA, so CUDA is the correct first backend — but the optimization logic itself must not contain CUDA-specific assumptions. Architect a **Compute Backend Interface** (CPU and CUDA implemented now; HIP/SYCL as a stated future backend) so the claim is "CUDA is our first acceleration backend, not our algorithmic dependency" rather than an implicit second foreign-hardware lock-in.

---

## 4. System architecture

```
                        Industrial Model (.mps / custom DSL / API)
                                       │
                                       ▼
                        ┌───────────────────────────┐
                        │   PRAMAAN-IR (Canonical    │   Objective, variables, bounds,
                        │   Intermediate Repr.)      │   types, sparse A / Q, metadata
                        └─────────────┬─────────────┘
                                      ▼
                        ┌───────────────────────────┐
                        │  Reversible Presolve +     │   Fixed-var elimination, singleton
                        │  Transformation Ledger     │   substitution, bound tightening,
                        │  + Ruiz Scaling            │   Ruiz scaling — every step logged
                        └─────────────┬─────────────┘
                                      ▼
                        ┌───────────────────────────┐
                        │ Structural Fingerprint     │   Bipartite graph analysis →
                        │ Engine                     │   block/temporal structure,
                        └─────────────┬─────────────┘   numerical risk, strategy choice
                                      │
                         ┌────────────┴────────────┐
                         ▼                          ▼
                 Monolithic Path            Decomposed Path (Benders)
                         │                          │
                         ▼                          ▼
        ┌────────────────────────────────────────────────────────┐
        │        Heterogeneous Continuous Engine (LP/QP)          │
        │  CPU: Dual/Revised Simplex + Interior-Point (LP & QP),  │
        │       warm-start, FP64 polish                           │
        │  GPU: Restarted PDHG (FP32-first, Precision Ladder),    │
        │       batched block/scenario relaxations                │
        └─────────────────────────┬────────────────────────────┘
                                   ▼
                        ┌───────────────────────────┐
                        │  Discrete Engine (MILP)    │  Multi-core parallel branch-and-cut
                        │  Branch • Cut • Heuristic  │  (worker threads over tree nodes),
                        └─────────────┬─────────────┘  warm-started node LPs, Gomory/MIR
                                                        cuts, best-bound/hybrid node
                                                        selection, feasibility pump/RINS
                                      ▼
                        ┌───────────────────────────┐
                        │  Warm-Start / Incremental  │  Basis cache + delta detection
                        │  Resolve Engine            │  across successive rolling-horizon
                        └─────────────┬─────────────┘  re-solves
                                      ▼
                        ┌───────────────────────────┐
                        │  GPU Scenario-Parallel     │  Optional: batch Monte Carlo-
                        │  Engine (robustness mode)  │  perturbed scenarios for a
                        └─────────────┬─────────────┘  hedged/robust plan
                                      ▼
                        ┌───────────────────────────┐
                        │  Certificate Generator     │  Residuals, integrality check,
                        │                            │  bound gap, ledger reference
                        └─────────────┬─────────────┘
                                      ▼
                        ┌───────────────────────────┐
                        │  Independent Verifier      │  Separate binary, re-checks
                        │  (pramaan-verify)          │  from scratch, PASS/FAIL
                        └─────────────┬─────────────┘
                                      ▼
                            VERIFIED SOLUTION + CERTIFICATE
```

---

## 5. Technology stack

- **Core solver:** C++ (performance, memory control, matches the PS's implicit expectation for a serious numerical engine)
- **Multi-core CPU parallelization** (explicit PS requirement, distinct from GPU acceleration): thread pool over the branch-and-cut tree (parallel node evaluation), parallelized presolve passes where independent — implemented with standard C++ threads/OpenMP, no GPU involved
- **GPU layer:** CUDA (first backend), abstracted behind a Compute Backend Interface for future HIP/SYCL support
- **Sparse storage:** CSR/CSC formats throughout — never dense storage for the constraint matrix
- **Bindings/tooling:** Python bindings over the C++ core for the CLI, benchmark harness, and any demo tooling
- **Model I/O:** `.mps` parser (industry standard, required for MIPLIB/Netlib/QPLIB compatibility) plus an optional lightweight native modeling API
- **No dependency, direct or indirect, on:** HiGHS, SCIP, CBC, GLPK, OR-Tools, CPLEX, Gurobi, or any other existing solver's internals. These may only be used externally for benchmarking comparisons, never inside the solve path.

---

## 6. Benchmark & validation strategy

1. **Baseline credibility (expected by judges):** cold-start solve time vs. HiGHS on Netlib LP, a MIPLIB 2017 benchmark subset, and QPLIB instances. Report honestly — being slower than mature commercial/open-source engines while being sovereign, inspectable, and verifiable is a legitimate and credible result; do not claim to "beat Gurobi/CPLEX" without an instance-level number backing it.
2. **Numerical robustness stress test (explicit PS requirement — do not skip this one):** a curated set of degenerate, weakly-conditioned, and ill-conditioned instances (from Netlib/Mittelmann's known-hard subsets, or synthetically constructed cases with extreme coefficient ranges, e.g. 10⁻⁸ to 10⁹ mixed in one matrix), demonstrating reliable convergence where naive/simpler implementations stall, cycle, or fail — reported via Ruiz-scaling before/after conditioning numbers and the certificate's residual trace. This directly answers the PS's "clear demonstration of numerical robustness" clause and should get its own slide, not be folded into the general benchmark slide.
3. **Flagship differentiated benchmark:** cumulative time-to-solution over a simulated rolling 30-day refinery rescheduling trace, cold-restart-every-time vs. PRAMAAN warm-start.
4. **Precision Ladder benchmark:** time/residual/energy tradeoff plot across CPU-FP64-only vs. GPU-FP32-only vs. PRAMAAN's adaptive ladder.
5. **Live trust demo:** solve → valid certificate → deliberately tamper with the solution → independent verifier rejects it in real time.
6. **Flagship industrial case:** a crude-blending + multi-period production scheduling model (combined, not just static blending) built with real-ish assay/capacity/demand data, sized realistically (tens of variables for the idea/PPT round demo model, scaled up for the finale if time allows).

---

## 7. Phased scope (discipline against over-scoping)

**P0 — Must genuinely work (idea/PPT round + early finale hours):**
Custom `.mps` parser · sparse CSR/CSC · reversible presolve + Ruiz scaling · CPU LP core (revised/dual simplex **and** interior-point) · GPU PDHG LP core (FP32-first) · automatic CPU/GPU path selection · multi-core parallel branch-and-cut skeleton · residual measurement · degeneracy/ill-conditioning stress-test set · independent LP verifier · benchmark runner vs. HiGHS on Netlib subset.

**P1 — Makes the submission distinctive (finale core deliverable):**
Structural fingerprinting + basic block detection · Precision Ladder · Transformation Ledger · MILP branch-and-cut with warm-started nodes · certificate format (LP + MILP) · warm-start/incremental resolve engine · rolling-horizon benchmark trace.

**P2 — Advanced / stretch / roadmap (present as roadmap, not claimed as built):**
Full Benders decomposition · GPU scenario-parallel robustness engine · QP active-set engine · exact/rational-arithmetic certificate verification · HIP/SYCL backend · multi-GPU scaling.

**Explicitly do not attempt to fully build during SIH:** MIQP, NLP, MINLP, distributed solving, GPU-native simplex/IPM, a large cut-family library, or a full modeling DSL. The PS itself treats MIQP/NLP/MINLP as future extensions, not initial deliverables — say this explicitly in the roadmap slide rather than overclaiming scope.

---

## 8. USP summary (for the pitch deck)

**One-line USP:** *"Compute fast. Polish precisely. Prove independently."*

**Sub-line:** *A structure-aware, precision-adaptive optimization engine that produces proof-carrying industrial decisions — not just fast answers.*

Four promises, stated plainly:

- **STRUCTURE** — exploit the mathematics hidden inside industrial models instead of treating them as generic sparse matrices.
- **SPEED** — heterogeneous CPU/GPU execution and warm-started incremental resolve, designed around how MRPL actually operates (repeated rolling-horizon re-solves), not just cold-start benchmark speed.
- **PRECISION** — spend numerical precision only where it's actually required, appropriate to real (consumer-GPU) hardware.
- **TRUST** — every critical optimization result ships with an independently verifiable certificate.

**Do not claim in the PPT** (credibility risks flagged explicitly): "AI automatically selects the best algorithm" (generic, everyone will have this slide), "our GPU solver is N× faster" without a measured number, "beats Gurobi/CPLEX" without an instance-level benchmark, or PDHG-on-GPU itself as the novelty (it's well-established prior art via Google's PDLP and cuPDLP.jl — cite it as informed foundation, not as your innovation).

---

## 9. Feasibility

- **Technical:** every P0/P1 component is either standard numerical methods (simplex, presolve, scaling, branch-and-cut) or a lightweight control-flow layer on top of them (precision ladder, warm-start cache, certificate generation) — not new research. The one genuinely novel-for-hackathon piece (structural decomposition + scenario-parallel GPU batching) is scoped as P1/P2 and framed honestly as a roadmap item where full generality isn't achieved.
- **Operational (team hardware):** consumer GPUs (not HPC-grade FP64 accelerators) directly motivate and justify the Precision Ladder's FP32-first design — this isn't a compromise, it's the correct engineering decision for the actual hardware, and should be stated as such to judges.
- **Financial:** no paid infrastructure required for P0/P1 — CPU dev work is fully local; GPU kernels can be developed/tested in isolated modules and benchmarked on available consumer GPUs or short cloud GPU bursts if needed for the finale.
- **Judging risk mitigation:** every claim in the deck is paired with either a measured benchmark or an explicit "roadmap, not yet built" label — this is itself part of the credibility strategy, since overclaiming (unverified speedups, "beats Gurobi") is a known way to lose judge trust.

---

## 10. Terminology reference (keep consistent across all slides, code, and docs)

| Term                                    | Meaning                                                                                       |
| --------------------------------------- | --------------------------------------------------------------------------------------------- |
| PRAMAAN-IR                              | The canonical internal model representation every input is converted into                     |
| Transformation Ledger                   | Log of every reversible presolve step, used to reconstruct/verify the original-space solution |
| Structural Fingerprint                  | The report describing a model's block/temporal structure and numerical risk                   |
| Precision Ladder                        | FP32-GPU-fast-pass → FP64-CPU-polish-if-needed execution strategy                            |
| Warm-Start / Incremental Resolve Engine | Basis-cache + delta-detection layer for repeated rolling-horizon re-solves                    |
| Scenario-Parallel Engine                | GPU batching of Monte Carlo-perturbed problem instances for robustness                        |
| Certificate                             | The residual/integrality/bound-gap proof object emitted alongside a solution                  |
| `pramaan-verify`                      | The independent, separately-implemented verifier binary                                       |

---

## 11. PS traceability matrix — every requirement, mapped

This table exists so nobody on the team, or reviewing this doc, has to take the architecture's compliance on faith. Every clause is taken directly from the official PS26119 text.

| PS clause (verbatim intent)                                                                              | PRAMAAN component                                                                                                                                                           | Status                                         |
| -------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------- |
| LP, MILP, QP initial focus; MIQP/NLP/MINLP later via modular design                                      | LP/QP continuous engine + MILP discrete engine now; §7 roadmap names MIQP/NLP/MINLP as explicit future extensions                                                          | Core (P0/P1)                                   |
| Revised simplex**and** interior-point methods                                                      | Both implemented from scratch on CPU (§2 Pillar 2, §4 diagram)                                                                                                            | Core (P0)                                      |
| Branch-and-bound / branch-and-cut / cutting planes / presolve / heuristics / advanced node selection     | Discrete Engine: parallel branch-and-cut, Gomory/MIR cuts, feasibility pump/RINS, best-bound/hybrid node selection (§4)                                                    | Core (P0/P1)                                   |
| Sparse matrix techniques, efficient numerical linear algebra                                             | CSR/CSC throughout, never dense (§4, §5)                                                                                                                                  | Core (P0)                                      |
| Multi-core parallelization                                                                               | Explicit thread-pool over the B&B tree, parallel presolve (§5)                                                                                                             | Core (P0)                                      |
| GPU acceleration "where it provides measurable benefit" (not mandated everywhere)                        | GPU used only where genuinely parallel: PDHG for large sparse LPs, decomposed blocks, scenario batches — never forced onto inherently sequential/irregular work (§2, §3) | Core (P0/P1), honestly scoped                  |
| Not built upon any existing open-source solver library                                                   | Explicit no-dependency clause (§1, §3, §5)                                                                                                                               | Core, non-negotiable                           |
| Scalability to thousands–millions of variables/constraints                                              | Sparse storage, presolve, structural decomposition for genuinely large models (§2 Pillar 1)                                                                                | Core (P1), decomposition applied conditionally |
| Solve MIPLIB/Netlib/Mittelmann benchmark problems                                                        | Benchmark plan item 1 (§6)                                                                                                                                                 | Core (P0)                                      |
| Compare vs. ≥1 established commercial/open-source solver                                                | HiGHS comparison (§6)                                                                                                                                                      | Core (P0)                                      |
| **Explicit demonstration of numerical robustness** on degeneracy/weak relaxations/ill-conditioning | Dedicated stress-test benchmark, own slide (§6 item 2)                                                                                                                     | Core (P0) — do not skip                       |
| Basic API/CLI sufficient; GUI not required                                                               | Python bindings + CLI only; no GUI built or promised anywhere in scope (§5)                                                                                                | Core, matches exactly                          |
| Refinery scheduling, crude blending, production planning, supply chain from open literature              | Flagship demo: combined crude-blending + multi-period scheduling model (§6 item 6)                                                                                         | Core (P0/P1)                                   |
| Transparent, extensible, sovereign foundation for future work                                            | Compute Backend Interface for future HIP/SYCL, MIQP/NLP/MINLP roadmap, reversible Transformation Ledger (§3, §7)                                                          | Supports the "extensible" clause directly      |

Everything in this table is either **Core** (directly requested by the PS text) or explicitly labeled **bonus/differentiator** elsewhere in this doc (Pillar 3, Pillar 4's exact-rational-arithmetic stretch, decomposition's conditional structural path). Nothing in the architecture exists that isn't either satisfying a literal PS clause or clearly flagged as an optional layer built on top of a completed core — this is the check against over-scoping into something a judge could read as unrelated to what was actually asked.

---

## 12. Instructions for AI assistants helping this team

If you are an AI assistant (Claude, ChatGPT, Gemini, or otherwise) being asked to help this team with slides, code, benchmarks, or further ideation on this project: **this document is the finalized architecture.** Work within its four pillars, phased scope, and terminology. If a request seems to call for a different core algorithm, a different headline USP, or scope beyond what P0/P1/P2 define, say so explicitly and flag it for the team to decide, rather than silently substituting your own architecture — the team has deliberately synthesized and pruned three independent AI proposals to reach this design, and inconsistent outputs across team members' tools is the specific failure mode this document exists to prevent.
