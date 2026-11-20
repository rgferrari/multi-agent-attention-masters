# CLAUDE.md — agents/

Documentation for the MARL algorithm implementations in this directory. See the root [CLAUDE.md](../CLAUDE.md) for project-wide setup, commands, and evaluation/metrics.

## Algorithm Implementations

**MAAC** ([maac/](maac/)):
- [attention_sac.py](maac/attention_sac.py) — top-level `AttentionSAC` class; wraps per-agent policies and a shared attention critic
- [utils/critics.py](maac/utils/critics.py) — multi-head scaled dot-product attention critic; each agent encodes its own state-action pair and attends to all others
- [utils/agents.py](maac/utils/agents.py) — per-agent policy with discrete action output
- [utils/buffer.py](maac/utils/buffer.py) — circular replay buffer using numpy object arrays (handles variable-length obs per agent)
- Discrete action spaces only. Attention entropy is regularized alongside policy entropy (SAC-style reward scaling).

**MAAC Continuous** ([maac_continuous/](maac_continuous/)):
- [attention_sac_continuous.py](maac_continuous/attention_sac_continuous.py) — `AttentionSACContinuous`; same structure as MAAC but for continuous spaces
- [utils/policies.py](maac_continuous/utils/policies.py) — `GaussianPolicy`: mean + log_std, tanh squashing, reparameterization trick
- [utils/critics.py](maac_continuous/utils/critics.py) — `AttentionCriticContinuous`: scalar Q output; Q head receives the agent's own SA encoding (not state-only) so the reparameterization gradient flows through it
- Policy update is per-agent: agent `i` uses its reparameterized action; others use buffer actions (no cross-agent gradient leakage)

**MADDPG** ([maddpg_torch/](maddpg_torch/)):
- [maddpg.py](maddpg_torch/maddpg.py) — top-level `MADDPG` class
- [models.py](maddpg_torch/models.py) — actor (logits for discrete / tanh for continuous) and critic (concatenated global obs+actions) networks
- [buffer.py](maddpg_torch/buffer.py) — fixed-size float32 circular replay buffer
- Supports both discrete (Gumbel-softmax exploration) and continuous (Gaussian noise) action spaces.
- Note: [maddpg/](maddpg/) (without `_torch`) is an unused vendored copy of the original OpenAI MADDPG reference implementation (TensorFlow 1 / Python 3.5 style) — nothing in this codebase imports it; don't confuse it with `maddpg_torch/` above.

**MAPPO** ([mappo/](mappo/)):
- [mappo.py](mappo/mappo.py) — top-level `MAPPO` class; per-agent actor + per-agent *centralized* critic (fed the concatenated global obs, value-only). A single class handles both discrete and continuous, branching on `ActionSpec` (like MADDPG) — no separate continuous variant.
- [models.py](mappo/models.py) — `Actor` (Categorical for discrete / diagonal Gaussian with state-independent `log_std` for continuous) and `Critic` (scalar V(s)). Reuses `ActionSpec`/`mlp` from `maddpg_torch/models.py`.
- [buffer.py](mappo/buffer.py) — `RolloutBuffer`: on-policy, single rollout, per-agent typed arrays; `compute_returns` does per-agent GAE, `agent_minibatch_generator` yields per-agent shuffled minibatches (no cross-agent flattening) so heterogeneous obs/action dims work.
- Per-agent independent networks (not upstream MAPPO's parameter sharing) and feedforward-only (no recurrent policy). Single-env rollouts (no vectorized threads), so batches are one episode's worth of steps.

**HAPPO** ([happo/](happo/)):
- [happo.py](happo/happo.py) — `HAPPO`/`HAPPOConfig`. **Subclasses `MAPPO`**: networks (per-agent actor + per-agent centralized critic), rollout, and checkpointing are inherited unchanged; only `update()` is overridden.
- The whole algorithm is that override: agents are updated **sequentially in a random order** (`torch.randperm`), and each agent's surrogate is scaled by a `factor` accumulating how much the already-updated agents moved — `factor *= exp(new_log_prob - old_log_prob)`. Agent `i` therefore optimises against teammates' *post-update* behaviour, restoring the monotonic improvement guarantee MAPPO lacks.
- Reuses `agents/mappo/buffer.py`'s `RolloutBuffer`, which carries `factor_buf` (all-ones for MAPPO, so MAPPO's behaviour is unchanged).
- Logs `losses/factor_mean` in addition to MAPPO's tags — a diagnostic for how far the sequential update drifts from 1.0.

**MAT** ([mat/](mat/)):
- [mat.py](mat/mat.py) — top-level `MAT`/`MATConfig`; one *shared* encoder-decoder transformer for all agents (parameter count is independent of agent count), exposing the same `select_actions`/`get_values`/`update` interface as `MAPPO`
- [models.py](mat/models.py) — `SelfAttention` (causal masking optional), `EncodeBlock`, `DecodeBlock`, `Encoder` (unmasked; its value head is the centralized critic), `Decoder` (causally masked over the agent axis)
- [transformer_act.py](mat/transformer_act.py) — the autoregressive path (rollout: one decoder pass per agent) and the parallel path (update: whole sequence scored in one pass, equivalent via the causal mask). Verified to return identical log-probs.
- [buffer.py](mat/buffer.py) — `MATRolloutBuffer`: agent-stacked `(T, n_agent, dim)` arrays; minibatches are whole timesteps (all agents together), since agents cannot be scored independently
- Handles discrete and continuous, and heterogeneous obs/action dims via padding + action masking (an extension beyond published MAT — see Provenance above).
- Note: `Multi-Agent-Transformer/` is a gitignored clone of the upstream reference, present only if cloned locally — nothing imports it; don't confuse it with `mat/` above. See [Upstream sources](#upstream-sources) to re-clone at the pinned revision.

## Provenance: reference implementations and the adaptations made

Every algorithm here derives from a published reference implementation. This section records, per algorithm, **what the original is**, **what it was built for**, **what was changed**, and **why** — so a deviation from published behaviour can be traced to its motivation rather than mistaken for a faithful port.

Two distinct kinds of relationship appear below, and they are not interchangeable:

- **Vendored-and-modified** — upstream source lives in this repo and is executed directly, with local edits applied in place (`maac/`).
- **Reimplemented-from-reference** — the reference is read as a specification and a fresh implementation written against this repo's conventions; the reference is either vendored unused (`maddpg/`) or cloned outside version control (`on-policy/`, `Multi-Agent-Transformer/`, both gitignored). This covers `maddpg_torch/`, `mappo/`, `happo/`, and `mat/`.

### Upstream sources

The table below is the authoritative record of where each implementation came from. **The two gitignored clones are consultation-only and may be deleted at any time** — nothing imports them, and this table is what allows them to be re-cloned at the exact revision the port was written against. File/line citations elsewhere in this document refer to these revisions.

| Local dir | Upstream | Pinned revision | Status on disk |
|---|---|---|---|
| `maac/` | https://github.com/shariqiqbal2810/MAAC | `6174a01` (upstream HEAD; recovered by diff, see below) | **committed, executed + locally modified** — not deletable |
| `maddpg/` | https://github.com/openai/maddpg | `e977d75` (recovered by diff — exact match on all 4 files) | committed, unused (TF1) |
| `on-policy/` | https://github.com/marlbenchmark/on-policy | `de66d7a` | gitignored clone, unused |
| `Multi-Agent-Transformer/` | https://github.com/PKU-MARL/Multi-Agent-Transformer | `be3ff49` | gitignored clone, unused (~164M) |
| `maac_continuous/` | **none — no upstream exists** | n/a | **original derived work**, see below |

**`maac_continuous/` has no upstream reference, and correctly so: there is no published continuous MAAC.** Iqbal & Sha's paper and implementation are discrete-only by construction — the critic emits one Q per action and `gather`s by action index, which requires an enumerable action set. `maac_continuous/` was written by deriving from the vendored `maac/`, not by porting another source.

This matters for how MAMuJoCo results are framed. `maddpg_torch/`, `mappo/`, and `mat/` are ports of published algorithms, so any divergence is a *deviation from a reference*. `maac_continuous/` is **original derived work**: there are no published numbers to compare against, and it should be described as "a continuous adaptation of MAAC" rather than "MAAC on MAMuJoCo". The design decisions it embodies (Gaussian policy, scalar Q on the state-action encoding, per-agent reparameterized update) are choices made here, not inherited from a reference — see the MAAC section below for each one and its justification.

`maac/` and `maddpg/` were vendored without their `.git`, so their revisions were **reconstructed** by diffing the vendored files against upstream history rather than read from a pinned checkout:

- **`maddpg/` → `e977d75`** — unambiguous: `trainer/maddpg.py`, `trainer/replay_buffer.py`, `common/distributions.py`, and `common/tf_util.py` all match that commit byte-for-byte. The copy is unmodified upstream code.
- **`maac/` → `6174a01`** (upstream HEAD) — inferred, not byte-exact, because the vendored copy is *modified*. Ignoring the `agents.maac.*` import rewrite, `utils/agents.py`, `utils/policies.py`, `utils/buffer.py`, and `utils/make_env.py` are identical to HEAD; the files that differ are exactly the ones changed locally (`attention_sac.py`, `utils/misc.py`, `utils/critics.py` in commit `5744526`, plus `utils/env_wrappers.py`). Since `buffer.py` has had no upstream commits since `1006cff`, HEAD is the best-supported origin but cannot be proven to the commit.

One local `maac/` edit is **not** covered by commit `5744526` and is undocumented elsewhere — [utils/env_wrappers.py](maac/utils/env_wrappers.py) replaces `np.stack(obs)` with `np.array(obs, dtype=object)` in three places, so that per-agent observations of differing length can be batched without forcing a rectangular array (the same object-array approach `utils/buffer.py` uses). It is inert for this repo's MPE2 runs, which do not use these vectorized wrappers.

To restore a deleted clone for consultation (into a scratch dir — do **not** clone over `agents/maac/`, which carries local modifications):

```bash
# The two gitignored reference clones
git clone https://github.com/PKU-MARL/Multi-Agent-Transformer.git agents/Multi-Agent-Transformer
git -C agents/Multi-Agent-Transformer checkout be3ff49

git clone https://github.com/marlbenchmark/on-policy.git agents/on-policy
git -C agents/on-policy checkout de66d7a

# The two vendored originals, for diffing against the vendored copies
git clone https://github.com/openai/maddpg.git /tmp/ref-maddpg && git -C /tmp/ref-maddpg checkout e977d75
git clone https://github.com/shariqiqbal2810/MAAC.git /tmp/ref-maac && git -C /tmp/ref-maac checkout 6174a01
```

To re-derive what was changed locally in `maac/` (the import rewrite is expected noise):

```bash
diff <(git -C /tmp/ref-maac show 6174a01:utils/critics.py | grep -v '^from \|^import ') \
     <(grep -v '^from \|^import ' agents/maac/utils/critics.py)
```

A recurring justification across all four is **comparability**: this repo is a comparison study, so an adaptation that would make one algorithm's results incomparable to the others' is rejected even when it is faithful to that algorithm's own reference. See [RESEARCH_LOG.md](../RESEARCH_LOG.md) for dated entries on individual decisions.

### MAAC — `maac/` (vendored, modified in place) → `maac_continuous/` (reimplemented)

**The original.** [MAAC](https://github.com/shariqiqbal2810/MAAC) (Iqbal & Sha, 2019, *"Actor-Attention-Critic for Multi-Agent Reinforcement Learning"*). A SAC-style off-policy actor-critic where the distinguishing component is a **shared multi-head attention critic**: each agent encodes its own state-action pair and attends over every other agent's encoding, so the critic learns *which* teammates matter per state rather than consuming a fixed concatenation of all of them. Built for the discrete-action MPE particle environments.

**What it consists of.** `attention_sac.py` (top-level `AttentionSAC` wrapper), `utils/critics.py` (the attention critic), `utils/agents.py` + `utils/policies.py` (per-agent discrete policies), `utils/buffer.py` (replay buffer), plus upstream training scaffolding (`utils/make_env.py`, `utils/env_wrappers.py`) that this repo does not use — its training loop is [scripts/train/maac.py](../scripts/train/maac.py) instead.

**Adaptations to `maac/` itself** (it is vendored *and* executed, so these are edits to upstream source — commit `5744526`):

| Adaptation | Justification |
|---|---|
| Attention-magnitude regularization strength made a config key (`attend_mag_reg`) instead of hardcoded `1e-3` | It is a hyperparameter of the attention mechanism under study; hardcoding it prevents varying the exact quantity this thesis examines. |
| Attention entropy logged as one scalar (mean over heads) instead of a per-head `add_scalars` dict | Upstream's per-head dict does not aggregate across runs in tensorboard; a single scalar makes attention breadth comparable across environments and seeds. |
| Device derived from the input tensor instead of a `use_cuda` flag + `Variable` wrapper | `Variable` is deprecated in modern PyTorch, and the flag desynchronized from where tensors actually lived. |
| Explicit `sum(dim=2)` over the attention axis instead of `squeeze()` | `squeeze()` silently collapses the batch axis when `n_other_agents == 1` (any 2-agent env, e.g. `simple_speaker_listener_v4`), corrupting the entropy value. A correctness fix, not a preference. |

**`maac_continuous/` — the discrete→continuous adaptation.** MAAC is discrete-only: its critic emits one Q per action (`odim` outputs) and `gather`s the taken action's index, which requires an enumerable action set. MAMuJoCo is continuous, so a separate class was written rather than branching the vendored code.

| Adaptation | Justification |
|---|---|
| `GaussianPolicy` (mean + state-dependent `log_std`, tanh squashing, reparameterization) replaces the categorical policy | Continuous control needs a density over ℝⁿ; tanh squashing bounds actions to the env's range, and the reparameterization trick is what makes the SAC policy gradient low-variance. |
| Critic Q head takes the agent's **own state-action encoding** (`sa_encodings[i]`) and emits a **scalar**, where discrete MAAC takes the state-only encoding (`s_encodings[i]`) and emits one Q per action | With a continuous action there is no index to `gather`, so Q must be a function of the action itself. Feeding the SA encoding is also what lets the reparameterization gradient reach the policy through the critic — a state-only input would cut that path. |
| Policy update is strictly per-agent: agent `i` passes its reparameterized action while all others pass buffer actions | Prevents cross-agent gradient leakage — only agent `i`'s parameters should be updated by agent `i`'s policy loss. |
| Attention entropy logged per agent (`agent{i}/attention/mean_entropy`) rather than pooled | Matches the continuous critic's per-agent scalar Q heads; the discrete variant pools because its heads share one output structure. Noted because **the tensorboard tag path differs between the two variants**. |
| Kept as a separate class instead of branching `AttentionSAC` | The critic's output structure differs fundamentally (scalar vs. per-action vector), so branching would have meant rewriting the vendored critic — losing the ability to diff `maac/` against upstream. |

### MADDPG — `maddpg/` (vendored, unused) → `maddpg_torch/` (reimplemented)

**The original.** [OpenAI MADDPG](https://github.com/openai/maddpg) (Lowe et al., 2017, *"Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments"*) — the paper that established the centralized-critic/decentralized-actor (CTDE) template the rest of this repo builds on. Each agent gets a deterministic policy plus a centralized critic conditioned on **all** agents' observations and actions.

**What it consists of.** TensorFlow 1 / Python 3.5-era code: `trainer/maddpg.py` builds static graphs via `p_train`/`q_train` closures, `common/tf_util.py` wraps session management, `common/distributions.py` handles action distributions, `trainer/replay_buffer.py` the replay memory. `agents/maddpg/` is a copy of this kept **for reference only — nothing imports it**.

**Why reimplemented rather than vendored-and-run.** TF1 is unavailable on this project's Python 3.10 toolchain and the rest of the repo is PyTorch. Running the original would have required a second incompatible runtime, making shared training loops, buffers, and tensorboard logging impossible. *(Rationale supplied by the user; recorded in RESEARCH_LOG.md 2026-06-03.)*

| Adaptation | Justification |
|---|---|
| Static TF graph closures (`p_train`/`q_train`) → eager PyTorch `nn.Module`s with explicit optimizer steps | Matches the rest of the codebase and makes the update step readable/debuggable inline. |
| One class handles discrete **and** continuous via `ActionSpec`, where the original branched through `make_pdtype` distributions | MPE2 is discrete, MAMuJoCo continuous; a single class lets both entry points share `_build_maddpg`. `ActionSpec`/`mlp` were later reused by `mappo/` too. |
| Discrete exploration = Gumbel-softmax + ε-greedy (`epsilon`); continuous = additive Gaussian noise (`expl_noise`) | Gumbel-softmax keeps the discrete action differentiable for the deterministic policy gradient — the original achieved this through its distribution abstraction, which the port does not carry. |
| The original's policy regularizer (`loss = pg_loss + p_reg * 1e-3`, penalizing squared policy logits) is **not** implemented | An intentional omission to flag: it damps logit magnitude and its absence may make discrete policies more prone to saturation. Left out to keep the update rule minimal; **re-add before attributing any discrete-env instability to MADDPG itself.** |
| Fixed-size float32 circular buffer, sized in transitions, replacing the original's list-based buffer sized as `batch_size * max_episode_len` | Preallocated arrays avoid per-step Python list growth; the episode-derived sizing didn't fit this repo's step-based `update_every`/`start_steps` cadence. |

### MAPPO — `on-policy/` (reference clone, gitignored) → `mappo/` (reimplemented)

**The original.** [marlbenchmark/on-policy](https://github.com/marlbenchmark/on-policy) (Yu et al., 2021, *"The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games"*) — the canonical MAPPO benchmark, showing on-policy PPO matches or beats off-policy MARL methods when properly tuned. The relevant algorithm is `r_mappo` (the `r` denotes **recurrent**).

**What it consists of.** ~79 Python files: `algorithms/r_mappo/r_mappo.py` (PPO trainer), `algorithm/r_actor_critic.py` (`R_Actor`/`R_Critic` with optional `RNNLayer`), `utils/shared_buffer.py` and `separated_buffer.py` (parameter-**shared** vs. per-agent variants), `utils/valuenorm.py`, plus a full runner/env-wrapper harness. It is a benchmark framework, not a single algorithm file. **Cloned for porting, not vendored** — it is gitignored (`.gitignore:7`).

| Adaptation | Justification |
|---|---|
| **Per-agent independent networks**, not upstream's parameter sharing (`shared_buffer.py` / shared runner) | Parameter sharing requires homogeneous agents; 5 of this repo's 6 MPE2 envs are heterogeneous (differing obs/action dims). It also keeps MAPPO structurally aligned with MADDPG/MAAC here, so the comparison isolates *critic conditioning* rather than confounding it with weight sharing. |
| **Feedforward only** — `RNNLayer` and the recurrent/naive-recurrent code paths dropped | This repo's MPE2 episodes are 25 steps with near-full observability; recurrence mainly benefits long-horizon partial observability (SMAC). Dropping it removes the `rnn_states`/`data_chunk_length` machinery threaded through the reference's buffer and trainer. |
| **Single-env rollouts**, not the reference's 32 vectorized `n_rollout_threads` | Matches the single-env stepping the MAAC/MADDPG loops already use, so all algorithms see the same env interface and episode accounting. Cost: each PPO update sees one episode's steps, a much smaller batch than upstream — relevant when comparing sample efficiency. |
| **Value normalization (`ValueNorm`) and PopArt dropped**; huber loss dropped in favour of clipped MSE | Both are variance-reduction utilities orthogonal to the CTDE question under study. Their absence may cost raw performance versus published MAPPO numbers — worth stating if these results are compared against the paper's. |
| `agent_minibatch_generator` yields **per-agent** minibatches with no cross-agent flattening; advantages normalized per agent | The reference flattens agents into the batch axis, which presumes uniform shapes. Per-agent generation is what allows heterogeneous obs/action dims. |
| Reuses `ActionSpec`/`mlp` from `maddpg_torch/models.py` | One discrete/continuous abstraction across both algorithms; avoids a second, divergent action-space representation. |

### HAPPO — `on-policy/onpolicy/algorithms/happo/` (reference clone) → `happo/` (reimplemented)

**The original.** [HAPPO](https://github.com/marlbenchmark/on-policy) (Kuba et al., 2022, *"Trust Region Policy Optimisation in Multi-Agent Reinforcement Learning"*, ICLR 2022), shipped inside the same `on-policy` benchmark repo as MAPPO. HAPPO addresses a real gap in MAPPO: when all agents update simultaneously, each optimises as though its teammates were frozen, so the joint update carries no monotonic improvement guarantee. HAPPO applies the **multi-agent advantage decomposition theorem** to update agents one at a time.

**What it consists of.** `algorithms/happo/happo_trainer.py` (231 lines) and `algorithms/happo/policy.py` (135 lines) — but the trainer is nearly a copy of `r_mappo.py`; the actual algorithm lives in `runner/separated/base_runner.py:135-182`, which owns the sequential loop and the factor accumulation.

| Adaptation | Justification |
|---|---|
| Implemented as a **subclass of `MAPPO`** overriding only `update()` | HAPPO differs from MAPPO solely in the update procedure — same networks, same rollout, same checkpoint format. Subclassing makes that relationship explicit in code and guarantees the two are compared on identical architecture, which matters for a study isolating *what the theorem buys*. |
| `factor` lives in `agents/mappo/buffer.py` as `factor_buf`, all-ones by default | Avoids a near-duplicate buffer. MAPPO never sets it, so its behaviour is bit-identical to before (verified). |
| Log-probs for the factor are evaluated over the **whole rollout in one pass**, not per minibatch | The factor compares policy behaviour before vs. after an agent's full update; computing it per minibatch would mix pre- and post-update states. |
| Dropped `PopArt`/`ValueNorm`, huber loss, active-masks, and the recurrent path | Same rationale as the MAPPO and MAT ports — consistency across the three PPO-family algorithms so differences trace to the algorithm, not to which utilities each port happened to keep. |
| Random permutation each iteration (`torch.randperm`), as upstream | A fixed order would systematically privilege whichever agent updates first. |

**Why HAPPO is interesting here.** It and MAT rest on the *same theorem* but apply it in opposite places: HAPPO sequentialises the **optimisation** (one policy at a time), MAT sequentialises the **architecture** (autoregressive decoder). MAPPO applies it nowhere. That makes the three a clean ablation of one idea, which is a stronger thesis argument than comparing four unrelated algorithms.

### MAT — `Multi-Agent-Transformer/` (vendored, unused) → `mat/` (reimplemented)

**The original.** [PKU-MARL/Multi-Agent-Transformer](https://github.com/PKU-MARL/Multi-Agent-Transformer) (Wen et al., 2022, *"Multi-Agent Reinforcement Learning is a Sequence Modeling Problem"*). MAT recasts MARL as sequence modeling: a single encoder-decoder transformer treats the **agent axis as the sequence axis**, and actions are emitted **autoregressively over agents** so agent `i` conditions on agents `0..i-1`'s already-chosen actions. That ordering is what supplies its multi-agent advantage decomposition and monotonic-improvement guarantee. PPO-trained.

**What it consists of.** ~140 files, built on the same `on-policy` framework skeleton: `algorithms/mat/algorithm/ma_transformer.py` (encoder/decoder/attention), `algorithms/utils/transformer_act.py` (autoregressive + parallel action paths), `algorithms/mat/mat_trainer.py` (PPO update), `utils/shared_buffer.py`, plus runners for SMAC/MPE/MAMuJoCo/Football/dexterous hands. Vendored **for reference only — nothing imports it**.

| Adaptation | Justification |
|---|---|
| **Heterogeneous space support**: observations zero-padded to `max_i obs_dim_i`, plus a per-agent action mask driving surplus logits to `-1e10` | The decisive adaptation. Upstream sizes *every* agent from `observation_space[0]` ([base_runner.py:68-91](https://github.com/PKU-MARL/Multi-Agent-Transformer/blob/be3ff49/mat/runner/shared/base_runner.py#L68-L91)), requiring uniform dims. Only `simple_spread_v3` qualifies among this repo's six MPE2 envs — a faithful port would run on **1 of 6**, making MAT incomparable to the other three algorithms. **This is an extension beyond published MAT and should be declared as such in the thesis.** Verified: masking holds under sampling (zero probability mass on invalid actions). |
| The `state_dim`/`encode_state` path was **removed**, not ported | It is dead code upstream: the reference overwrites the state with `np.zeros((..., 37))` before every use ([ma_transformer.py:238,257,281,304](https://github.com/PKU-MARL/Multi-Agent-Transformer/blob/be3ff49/mat/algorithms/mat/algorithm/ma_transformer.py#L238)), so the hardcoded `37` never carries information. Porting it would have replicated a bug. |
| Single shared optimizer over encoder+decoder with one combined loss, rather than MAPPO-style split actor/critic optimizers | Faithful to MAT: the encoder's value head *is* the critic, so the two cannot be optimized independently. This is the structural reason MAT logs model-wide loss scalars where MAPPO/MADDPG log per-agent ones. |
| `MATRolloutBuffer` stores agent-stacked `(T, n_agent, dim)` arrays with a single joint minibatch generator — no per-agent generator like `mappo/buffer.py` | Once the decoder conditions each agent on the others, agents cannot be scored independently; minibatches must contain whole timesteps with every agent present. |
| Dropped `ValueNorm`, huber loss, active-masks, and the recurrent `mat_gru.py` variant | Same rationale as MAPPO above — orthogonal utilities, and consistency with how MAPPO was ported keeps the two PPO-based algorithms comparable to each other. |
| Training loop mirrors `train_mappo` (same run-dir layout, episode tags, best-checkpoint/early-stop block) | MAT is PPO-trained, so reusing the established loop template keeps evaluation and checkpointing identical across algorithms. |

**Verification performed on the port** (beyond "it runs"): the autoregressive and parallel action paths return **identical** log-probs (max diff `0.0`) — the invariant that makes the PPO ratio valid, since rollout uses the former and the update the latter; action masking admits zero probability mass on padded logits; all six MPE2 envs train; the continuous/MAMuJoCo branch completes an update.

## CTDE: what each algorithm centralizes

All four follow **Centralized Training, Decentralized Execution (CTDE)**: the critic is **centralized** — it conditions on all-agent information during training. None of them use a "decentralized" (local-obs-only) critic.

Where they differ is *what the critic sees*, *whether it is instantiated per-agent or shared*, and — with MAT — whether agents even decide independently:

| | Actor | Critic instances | Critic input | Shares teammate *actions*? |
|---|---|---|---|---|
| **MADDPG** | per-agent, local obs | one **per agent** | global obs **+ all actions** → Q(s,a) | ✅ yes |
| **MAAC** | per-agent, local obs | one **shared** attention module (per-agent heads) | all agents' state-action pairs, via attention | ✅ yes |
| **MAPPO** | per-agent, local obs | one **per agent** | global obs only → V(s) | ❌ no (state-value baseline) |
| **HAPPO** | per-agent, local obs | one **per agent** | global obs only → V(s) | ❌ no (state-value baseline) |
| **MAT** | one **shared** transformer decoder over the agent sequence | the encoder's **value head** (no separate critic network) | all agents' observations, via unmasked self-attention → V(s) per agent | ⚠️ yes, but in the **actor** — agent `i` conditions on agents `0..i-1`'s actions autoregressively |

Two subtleties worth keeping distinct:

- **What the critic conditions on.** MADDPG and MAAC feed teammates' **actions** into the critic (action-value Q); MAPPO and MAT use a **state-value** V(s) that sees all observations but no actions — a defining property of PPO, not an omission.
- **Whether execution is truly decentralized.** MADDPG/MAAC/MAPPO agents each act from their own observation alone, so execution is decentralized in the strict sense. **MAT is the exception**: its autoregressive decoder means agent `i`'s action depends on agents `0..i-1`'s actions at *execution* time, not just during training. It is still CTDE in the paper's framing (no agent needs others' *observations*), but the independence assumption the other three share does not hold — relevant when comparing them as decision architectures.

## Key Patterns

**Device management** — call `prep_training()` before gradient updates and `prep_rollouts()` before inference; these move networks between devices and toggle `requires_grad`.

**Soft target updates** — `target = (1 - tau) * target + tau * source` used in both algorithms for stability.

**Checkpoint format:**
- MAAC saves `{'init_dict', 'agent_params', 'critic_params'}` so the full model can be reconstructed from file without a running instance
- MADDPG saves full `state_dict()` for actors, critics, and optimizers
- MAPPO saves full `state_dict()` for actors, critics, and optimizers (same pattern as MADDPG); reload by rebuilding via `_build_mappo` then `load_state_dict`
- MAT saves `{'encoder', 'decoder', 'optimizer', 'config'}` — one encoder/decoder pair and a single optimizer, not per-agent lists, since the model is shared across agents; reload by rebuilding via `_build_mat` then `load_state_dict`

**Action representation** — actions are passed around as `Dict[agent_id → int]` (discrete index) or `Dict[agent_id → np.ndarray]` (continuous) at environment boundaries, and converted to one-hot tensors inside training updates.
