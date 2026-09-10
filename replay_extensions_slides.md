# Replay Extensions — Presentation Outline

One slide per section. Each idea follows the same four beats:
**Proposal → Issue → Question → Hopeful outcome.**

---

## Slide 1 — The framing

- Replay buffer currently stores **224² fp32 tensors: 602 KB/exemplar, ~6.6 GB at 11k**
- Exemplar *count* has saturated (v2–v6): more of the same images stops helping
- So the question is no longer *how many*, but:

> **What does the replay buffer actually need to store for the replay loss to do its job?**

- Four ideas below change *what* is stored, not how much

---

## Slide 2 — Prerequisite: find the saturation knee

- **Proposal:** sweep the buffer budget *downward* — 500 / 1k / 2.75k
- **Issue:** budget has only ever gone up (5.5k → 11k → 13k), so every data point sits inside the saturated regime; the knee has never been located
- **Question:** at what budget does replay actually start to matter?
- **Hopeful outcome:** a number that sets the operating budget for every experiment below

- Zero code — existing flags. ~3 runs.
- **Has to run first:** the compression ideas can only pay off *below* the knee

---

## Slide 3 — Idea 1: Low resolution

- **Proposal:** store at 112² / 64² / 32², upsample back to 224 at replay time
  - 4× / 12× / 49× cheaper per exemplar
- **Issue:** CLIP was pre-trained at 224 with a fixed patch grid — low-res inputs are further off-distribution than they would be for a CNN
- **Question:** does storing 4× more images at half resolution beat 1× at full resolution, *at the same storage cost*?
- **Hopeful outcome:** coarse images are good enough for the replay CE term → 4–12× cheaper buffer at no accuracy cost

**Two comparisons, not one:**

| Comparison | Holds fixed | Answers |
|---|---|---|
| 1k @ 224² vs 1k @ 112² | count | What does low-res *cost*? |
| 1k @ 224² vs 4k @ 112² | storage | Do the extra images *pay for* it? |

- Runtime is **unchanged** — everything is upsampled to 224 before the encoder. Low-res buys storage, not speed.

---

## Slide 4 — Idea 2: Store features instead of images

- **Proposal:** store CLIP's 512-d embedding (1 KB) instead of pixels (602 KB) — ~600× smaller
- **Issue:** replay CE backprops through the image encoder; a stored embedding removes that path. Not a drop-in swap.
- **Question:** is the image encoder stationary enough under ZSCL + RD for stored features to stay valid?
- **Hopeful outcome:** ~600× smaller buffer *plus* a mechanistic explanation of why replay works at all

---

## Slide 5 — Why that issue is survivable

**Which loss updates which encoder** (verified in code):

| Loss | Image enc. | Text enc. |
|---|---|---|
| `L_ce` (current task) | yes | yes — *current* classes |
| `L_zscl` | yes | — |
| `L_RD` | yes | — |
| `L_rep` (replay CE) | yes | yes — **old** classes |

- Image encoder is **already anchored twice** (`L_zscl`, `L_RD`) — replay's image gradient is a third, weaker copy
- `L_rep` is the **only** term anchoring the text encoder to old classes — that is replay's unique job
- Feature replay drops the redundant half, keeps the unique half

---

## Slide 6 — De-risking Idea 2: the drift gate

- **Failure mode:** stored features go stale as the encoder moves → text encoder aligned to an image manifold the model no longer produces
- **Test:** re-encode the v4 buffer with the task-0 checkpoint and the final checkpoint; measure cosine similarity per task; compare against Seq-FT
- **Cosine > ~0.95** → feature replay is licensed, and it's a strong figure on its own
- **Cosine ~0.7** → drop the direction

- **Cost: one day, zero GPU hours, existing checkpoints.** Run before writing any code.

---

## Slide 7 — Idea 3: VQ tokens (the "gist" idea, modernised)

- **Proposal:** store each exemplar as a grid of VQ-GAN codebook indices (~320 bytes, ~1800×), decode back to pixels at replay time
- **Issue:** literal GIST can't work — it's a hand-crafted descriptor in a space CLIP has never seen, and can't be fed to the image encoder. VQ needs an external decoder, decode cost inside the training loop, and may produce off-distribution artifacts.
- **Question:** can a reconstruction carry enough class signal for the replay CE term?
- **Hopeful outcome:** extreme compression **while keeping the full gradient path through the image encoder** — fixes the one real weakness of feature replay

- Most novel item on the list, and the most work (~2 weeks)
- Recommend deferring until low-res and feature results are in

---

## Slide 8 — Idea 4: Class-name replay (zero storage)

- **Proposal:** drop stored images entirely; replay previous tasks' **class names** on the text side against the frozen teacher
  - Storage: 11 lists of strings, ~5 KB
- **Issue:** cross-entropy needs images to *place* a decision boundary, not just to name the classes — names alone may not be enough
- **Question:** what does storing an image of a Boeing 737 give us that the string `"a photo of a Boeing 737"` does not?
- **Hopeful outcome:**
  - Partial recovery → a free lower bound on how much of replay is available at zero storage
  - Full recovery → the headline result

- Distinct from ZSCL's text loss, which uses generic Conceptual Captions sentences, not old tasks' actual label sets
- Cheapest thing on the list to run (~3 days, ~60 lines)

---

## Slide 9 — Costs

Baseline: **~13.5 h per 11-task run** (measured, v4 on the 40 GB H100 MIG slice).
4-task protocol ≈ **4 h** (scaled by iteration count, not measured).

| Idea | Code | Code time | Runs (4-task) | GPU-h | Risk |
|---|---|---|---|---|---|
| Knee sweep | none (flags) | 0 | 3 | 12 | none |
| Low-res | ~15 lines | 0.5 day | 4 | 16 | low |
| Drift gate | analysis only | 1 day | 0 | ~0 | none |
| Class-name replay | ~60 lines | 3 days | 2 | 8 | medium |
| Feature replay | new buffer + loss | ~1 week | 5 | 20 | medium |
| VQ tokens | VQ-GAN + decode | ~2 weeks | 4 | 16 | high |

- **Sweep on 4-task, promote winners to 11-task** — 12 GPU-h to find out instead of 40

---

## Slide 10 — What I need decided

- **Positioning:** is the thesis's main contribution (a) the RD mechanism with storage as support, or (b) storage efficiency with RD as the method it's shown on?
  - Context: matched-baseline rebuttal analysis put RD at **−0.66 Last / +0.20 Transfer**, smaller than first reported
- **Protocol:** is 4-task acceptable for sweeps, or 11-task throughout? (~3.4× cost difference)
- **Storage fix:** adopt uint8 + replay-time augmentation? Frees 4× storage and enables JPEG, but breaks direct comparability with v3–v6
- **Scope:** which of the four ideas to actually pursue

---

## Suggested order of work

1. **Drift gate** — 1 day, no GPU. Gates Idea 2.
2. **Knee sweep** — no code, runs in background. Gates Idea 1.
3. **Low-res** — half a day of code, cheapest real experiment.
4. **Class-name replay** — 3 days, highest upside per hour.
5. **Feature replay** — ~1 week, conditional on the drift gate.
6. **VQ tokens** — only if compression looks like the interesting axis.

**Shortest path to a real result: drift gate + low-res ≈ 1.5 days of work and ~16 GPU-hours.**
