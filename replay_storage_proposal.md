# Thesis Extension Proposal: Rethinking What a Replay Buffer Stores

> **How to use this document.** Sections marked `[DECISION]` need input before work starts. Sections marked `[TODO]` are placeholders to fill in
> as results arrive. Everything else is a proposal and is open to revision.

---

## 1. Summary

Four extension directions were suggested plus two additions proposed here.
This document assesses each one for what it would contribute, what it would take to
build, and what it would cost to run.

Three of the options (resolution, gist, features) are *compression* schemes — they
change what is stored per exemplar rather than how many exemplars are stored. The
other options vary count or sampling and are reported in exemplars, per the standard
convention in the replay literature.

`[DECISION]` Which options to pursue. See §8 for costs and §9 for a proposed order.

---

## 2. Low resolution vs. high resolution

**Proposal.** Store exemplars at 112², 64², 32²; bilinear upsample to 224 at replay
time. Each exemplar becomes 4x / 12x / 49x cheaper to store.

**The actual question:** can we store *more* images at lower resolution and come out
ahead? A 112² exemplar costs a quarter of a 224² one, so the same storage buys 4x as
many. If coarse images are good enough for the replay CE term, that is a straight win.


| Comparison | Holds fixed | Answers |
|---|---|---|
| 1k @ 224² vs **1k @ 112²** | count | What does low-res *cost* per exemplar? |
| 1k @ 224² vs **4k @ 112²** | storage | Do the extra images *pay for* that cost? |

If the second wins, row 1 tells you whether it was because low-res barely hurts or
because quantity helped a lot.

**Why it should work.** The replay CE term only needs to prevent class decision
boundaries from drifting — a far weaker requirement than learning the task from
scratch. Coarse images plausibly preserve that.

**Why it is interesting for CLIP specifically.** Wang et al., *Memory Replay with
Data Compression* (MRDC, ICLR 2022) runs this experiment with JPEG quality on
standard class-incremental learning and finds the quantity gain outweighs the
fidelity loss. CLIP is the harder case: it was
pre-trained at 224 with a fixed patch grid, so low-resolution inputs are further
off-distribution than for a ResNet. **A result contradicting MRDC would be more
interesting than one confirming it**, and either outcome is publishable.

**Implementation.** Downsample in `ReplayBuffer.add_task`, resize in
`FlatReplayDataset.__getitem__`, add `--replay_store_res`. ~15 lines, half a day.

**Runtime.** Stored resolution does *not* change training cost — exemplars are
upsampled back to 224 before they hit the encoder, so the ViT sees the same input
shape and every iteration costs the same. Low-res buys storage, not speed. The only
added cost is a CPU resize in the dataloader (negligible) and a one-time buffer
rebuild per task (~minutes).

Three configurations are needed (1k @ 224², 1k @ 112², 4k @ 112²) — the 224² row is a
new baseline, since the budget has never been run below 5.5k.

| Protocol | Per run | 3 runs | Wall clock if run in parallel |
|---|---|---|---|
| 4-task (1,500 it./task ≈ 6k iters) | ~4 h | ~12 GPU-h | ~4 h |
| 11-task (v4 schedule, ~21.3k iters) | ~13.5 h | ~40 GPU-h | ~13.5 h |

Recommend sweeping on 4-task and promoting to 11-task only if the result is
interesting — 12 GPU-hours to find out, rather than 40. The ~13.5 h figure is
measured (v4 on the 40 GB H100 MIG slice); the 4-task figure is scaled from it by
iteration count, not measured directly.

**Known confound to fix while here.** The buffer currently stores *already-augmented*
tensors — one fixed crop per exemplar, frozen for the whole run. Downsampling
compounds this. Recommend storing raw resized uint8 and applying augmentation at
replay time. This is a fairer comparison *and* gives a free 4x saving from
fp32 → uint8 on top of the resolution saving.


---

## 3. Store features instead of images

**The strongest option, with one hard constraint.**

The replay CE at `src/models/training.py:815` calls `model(task_images, None)` — it
requires gradients through the entire image encoder. Storing final embeddings removes
that gradient path. Feature replay is therefore **not a drop-in substitution**; it
changes what the loss is able to update.

Two variants:

| Variant | Bytes/exemplar | Image-encoder gradient? | Verdict |
|---|---|---|---|
| Final 512-d embedding (fp16) | **1 KB** | No | Viable, see below |
| Hybrid: small pixel buffer + large feature buffer | tunable | Partial | Likely the best point |

**Why the no-gradient variant is still viable.** Losing the image-encoder gradient
sounds fatal, but the four loss terms do not overlap the way it first appears. Which
encoder each one actually updates.

| Loss term | Image encoder | Text encoder | Why |
|---|---|---|---|
| `L_ce` (current task) | yes | yes — *current* classes | both embeddings computed with grad (`training.py:534,538`) |
| `L_zscl` | yes | **no** | teacher text embeddings are under `no_grad` (`training.py:550-554`) |
| `L_RD` | yes | **no** | same structure, replay images instead of reference (`losses_phase3.py:45-49`) |
| `L_rep` (replay CE) | yes | yes — **old** classes | both computed with grad (`training.py:811,815`) |

Two consequences follow:

1. **The image encoder is already anchored twice without replay.** `L_zscl` distills
   the student against the frozen pre-trained CLIP on ImageNet reference images;
   `L_RD` does the same on replay images. Both compare student-vs-teacher similarity
   against a *fixed* set of CC text embeddings, and both backpropagate only through
   the student's image embeddings. Replay CE's image-side gradient is a third, weaker
   copy of a job two other terms are already doing.

2. **`L_rep` is the only term in the entire objective that anchors the text encoder to
   old classes.** Nothing else re-encodes previous tasks' class names with gradient
   attached. That is the job replay is uniquely doing, and it is what would be lost if
   replay were removed.

Feature replay drops exactly the redundant half and keeps exactly the unique half.
With a stored embedding the CE becomes `logits = stored_img_emb @ text_emb.T`, which
still backpropagates into `text_emb` — old-class text anchoring survives intact, and
only the image-side gradient, already covered twice, is given up.

**Why this is worth checking first.** Feature replay's known failure mode is
**representation drift** — stored features go stale as the encoder moves, so the text
encoder ends up being aligned to an image manifold the model no longer produces. Our
setup should be unusually resistant to it, precisely because the two terms in the
table that *do* update the image encoder, `L_zscl` and `L_RD`, both exist to hold it
against the frozen pre-trained weights.

That makes drift a *precondition we can measure cheaply before committing*. If the
encoder really is near-stationary, feature replay is licensed and we know **why** it
works rather than just that it does. If it isn't, we find out in a day rather than
after two weeks of implementation.

**Go/no-go gate (zero GPU cost, do this first).** Re-encode the v4 buffer with (a) the
task-0 checkpoint and (b) the final task-10 checkpoint. Measure cosine similarity
between the two feature sets, per task. Plot against a Seq-FT baseline.

- Cosine > ~0.95 under ZSCL+RD while Seq-FT degrades → feature replay is licensed,
  and this is a strong figure in its own right.
- Cosine ~0.7 → abandon the direction, two weeks saved.

`[TODO]` Drift measurement result: ______

**Implementation after the gate.** New `FeatureReplayBuffer`, feature-space CE path,
one-time encoding pass per task. ~1 week — the loss path genuinely differs rather
than being flag-gated.

---

## 4. Gist

**Taken literally it will not work.** GIST is a hand-crafted descriptor in a space
CLIP has never seen; it cannot be fed to the image encoder. Training a decoder from
GIST back to pixels would produce blurry texture fields with no object identity,
destroying exactly the class signal the CE term needs. **Recommend not pursuing the
literal version.**

**Two legitimate modern descendants:**

1. **CLIP features are the modern GIST** — a ~512-d learned global descriptor. This
   is Section 3, and it is the direct answer to what the suggestion is reaching for.

2. **VQ token replay** is closer to the literal intent. Encode each exemplar to a grid
   of VQ-GAN codebook indices (~256 tokens x 10 bits ≈ **320 bytes**, ~1800x
   compression), decode back to pixels with a frozen decoder at replay time.
   Store the gist, reconstruct the image. **Critically, this preserves the full
   gradient path through the image encoder** — it fixes the one real problem with
   feature replay.

   Most novel item on the list, and the most work: pretrained VQ-GAN dependency,
   decode cost inside the training loop, and reconstruction artifacts that may be
   off-distribution for CLIP. ~2 weeks.

`[DECISION]` Pursue VQ tokens? Recommend deferring until the low-res and feature
results show whether aggressive compression is where the interesting behaviour is.


---

## 5. Class-name replay — the zero-storage limit point

**The question only CLIP can ask.** The buffer is a label carrier. But CLIP already
knows the labels — the class names are in the text encoder's vocabulary. So: what
does storing an image of a Boeing 737 provide that the string
`"a photo of a Boeing 737"` does not?

**Proposal.** Replace image replay with a text-side term over previous tasks' class
names: self-distillation of old class-name embeddings against the frozen teacher.

**Storage: 11 lists of strings, ~5 KB** — effectively no buffer at all.

**Distinct from existing work.** ZSCL's text loss uses generic Conceptual Captions
sentences, not the previous tasks' actual label sets. This has not been tested.

**Honest expectation.** Likely recovers *some* but not all of the replay CE benefit —
cross-entropy needs images to place a decision boundary, not merely to name the
classes. But even a partial result is a valuable lower bound — it says how much of
the replay benefit is available for free — and a strong result would be the headline.

**Implementation.** ~60 lines, ~3 days. Cheapest possible thing to run.

`[TODO]` Class-name replay result: ______

---


## 6. Cost estimates

**Runtime baseline** (from `experiments_findings.readme`): ~13.5 h per full 11-task
run on the 40 GB H100 MIG slice. The v4 schedule is ~21,300 iterations; a 4-task
protocol at 1,500 iterations each is ~6,000 iterations ≈ **4 h**.

> **Key cost decision: run all sweeps on the 4-task protocol, confirm only the
> winning configurations on 11-task.** This is the difference between a two-week
> study and a two-month one.

| Option | Code required | Code time | Runs (4-task) | GPU-h | Risk | Payoff |
|---|---|---|---|---|---|---|
| Low-res replay | ~15 lines | 0.5 day | 4 | 16 | low | likely positive |
| Class-name replay | ~60 lines | 3 days | 2 | 8 | medium | large if it lands |
| Feature replay | new buffer + loss path | ~1 week | 5 | 20 | medium | the chapter's core |
| VQ token replay | VQ-GAN + decode loop | ~2 weeks | 4 | 16 | high | most novel |
| GIST (literal) | — | — | — | — | — | **do not pursue** |

---

## 7. Risk and open questions

**Positioning risk — needs an explicit decision.** This direction makes the thesis a
*storage-efficiency* contribution, which is a different pitch from the RD-mechanism
story in the current paper. Given that the rebuttal analysis found RD's effect is
smaller than originally reported (**−0.66 `Last` / +0.20 `Transfer`** against the
iteration-matched λ=0 baseline, versus the confounded −0.27 / +0.37), storage
efficiency may be the more robust place to put the thesis's weight.

`[DECISION]` Is the thesis's primary contribution (a) the RD mechanism, with storage
as a supporting chapter, or (b) storage efficiency, with RD as the method it is
demonstrated on? This should be settled before Week 3.

**Other open items:**

- `[DECISION]` 4-task protocol acceptable for sweeps, or does Faisal want 11-task
  throughout? (Cost difference is roughly 3.4x.)
- `[DECISION]` uint8 + replay-time augmentation change — adopt, given it breaks
  direct comparability with v3–v6?
- `[TODO]` Confirm Nibi allocation has headroom for ~90 GPU-hours over 5 weeks.
- Open question: does the fixed-crop-at-storage-time issue affect the v3–v6 results
  already reported? Worth checking before the paper is finalised.

---

## 8. Related work to cite

| Work | Relevance |
|---|---|
| Wang et al., *Memory Replay with Data Compression* (MRDC), ICLR 2022 | Direct precedent for the fidelity-vs-quantity tradeoff; the result to compare against |
| Oliva & Torralba, *Modeling the shape of the scene* (GIST), IJCV 2001 | Origin of the "gist" idea; cite when redirecting to learned descriptors |
| Rebuffi et al., *iCaRL*, CVPR 2017 | Herding-based exemplar selection; the count-side alternative |
| Zheng et al., *ZSCL*, ICCV 2023 | Base method |
| Yu et al., *MoE-Adapters*, NeurIPS 2024 | Current MTIL SOTA comparison |

`[TODO]` Add VQ-GAN citation if Section 4 is pursued.
