# Rebuttal runs for Reviewer zvRY (submission 31258)

Everything here is additive: no file under `mtil/src/` or
`mtil/scripts/4task/phase3/` is modified. The two control axes are installed at
run time by monkey-patching `phase3.trainer_phase3`'s module namespace
(`rd_controls/run_rd_control.py`), so the headline ExRD code path is byte-for-byte
the one used for the submitted numbers.

## What the reviewer asked, and which run answers it

| Reviewer zvRY | Run | Script | ~cost |
|---|---|---|---|
| Q1 "show when RD helps beyond standard replay; a per-task analysis" | iteration-matched λ=0 baseline, then the analysis script | `11task/rebuttal_replay_only_matched.sh` → `per_task_rd_analysis.py` | 13.5 h |
| Q1 "…or only on tasks close to CLIP pretraining" | measured CLIP zero-shot row, fed into the same analysis | `11task/rebuttal_zeroshot_eval.sh` | ~1 h |
| Q2 "same distillation loss on current-task images" | C1 | `11task/rebuttal_rd_current_images.sh` | 13.5 h |
| Q2 "…or references instead of replay images" | C3 | `11task/rebuttal_rd_reference_images.sh` | 13.5 h |
| Q3 "previous task checkpoint as teacher" | C2 | `11task/rebuttal_rd_prev_teacher.sh` | 13.5 h |

Every training run is `phase3_no_lora_11t_v4.sh` (the headline ExRD run) with a
single flag changed, so each is a clean single-factor control: same lr, label
smoothing, per-task iteration schedule, ZSCL branch, 11k proportional buffer,
replay CE weight, λ=0.3, and the same 10,599 Conceptual Captions anchors.

## Priority if GPU time is tight

1. `rebuttal_replay_only_matched.sh` — needed for Q1 **and** it fixes a confound
   in the current λ ablation (see below). Highest value.
2. `rebuttal_zeroshot_eval.sh` — cheap, and Q1's second half needs it.
3. `rebuttal_rd_prev_teacher.sh` (C2) — the reviewer's sharpest question; it
   tests the paper's central mechanistic claim head-on.
4. `rebuttal_rd_current_images.sh` (C1).
5. `rebuttal_rd_reference_images.sh` (C3) — lowest value, because RD on
   reference images with the frozen teacher *is* `L_ZSCL`; the run measures ZSCL
   at effective weight 1.3 plus replay. Worth stating that equivalence in the
   response rather than spending 13.5 h to rediscover it. Run it only if the
   reviewer's exact wording needs a literal answer.

C1, C2 and C3 are independent and can run concurrently on separate GPUs.

## The iteration-budget confound (read before writing the per-task answer)

The λ=0 row quoted in the response (86.44 Last / 68.00 Transfer) comes from
`ckpt/11task/ablation_prop_replay`, which was trained with a **uniform 1500
iters/task**. The ExRD run it is compared against uses the **per-task schedule**
(Aircraft 3000, StanfordCars 3000, SUN397 5000, MNIST 800, …). A per-task
breakdown between those two runs mixes RD's effect with an iteration-budget
effect, and the pattern shows it: the two largest apparent "RD gains" land
exactly on StanfordCars (3000 vs 1500 iters, +3.52) and SUN397 (5000 vs 1500,
+1.82), while the mean per-task ΔLast across the 11 trained tasks is −0.27.

`rebuttal_replay_only_matched.sh` removes the confound (v4 config, λ=0, nothing
else changed) and is what `per_task_rd_analysis.py` uses by default. Note that
it also produces an iteration-matched λ=0 row for the sensitivity table in the
Reviewer 3U1b response, which currently footnotes only three rows for altered
iteration counts.

## Running the training controls

On Nibi:

```bash
sbatch mtil/scripts/11task/rebuttal_replay_only_matched.sh
sbatch mtil/scripts/11task/rebuttal_zeroshot_eval.sh
sbatch mtil/scripts/11task/rebuttal_rd_current_images.sh
sbatch mtil/scripts/11task/rebuttal_rd_prev_teacher.sh
sbatch mtil/scripts/11task/rebuttal_rd_reference_images.sh
```

All five write to `ckpt/11task/rebuttal_*` (and `ckpt/rebuttal/zeroshot/`), and
all are resume-safe through phase3's existing checkpoint/completed-task logic.

To run a control by hand, take the ExRD command line verbatim and swap the
entry point plus one flag:

```bash
cd mtil
export PYTHONPATH=scripts/4task/phase3:scripts/rebuttal
python -m rd_controls.run_rd_control  <v4 flags...>  --rd_image_source current
python -m rd_controls.run_rd_control  <v4 flags...>  --rd_teacher prev_task
```

Control flags (everything else passes through to phase3 unchanged):

- `--rd_image_source {replay,current,reference}` — images the RD loss sees.
  `replay` (default) reproduces ExRD exactly.
- `--rd_teacher {frozen,prev_task}` — teacher for the RD term **only**; the ZSCL
  branch always keeps the frozen pre-trained teacher, so `prev_task` is a
  single-factor swap and not a different method. Caption embeddings come from
  the same checkpoint as the teacher, so the target is that teacher's own
  alignment distribution. Task 1 has no predecessor and no replay buffer, so RD
  is inactive there either way.
- `--rd_no_match_batch_size` — by default the RD image batch is subsampled to
  `--replay_batch_size` (8) for every source, so no control gets a larger
  distillation batch than ExRD. This flag disables that.

## Analysis

```bash
# Q1 per-task breakdown (after the matched baseline and zero-shot eval finish)
python mtil/scripts/rebuttal/per_task_rd_analysis.py \
    --zeroshot-csv mtil/ckpt/rebuttal/zeroshot/evaluate_all_results.csv \
    --out-dir mtil/ckpt/11task/rd_per_task

# Q2/Q3 control table (works with partial results; missing runs are labelled)
python mtil/scripts/rebuttal/rd_control_table.py --tex mtil/ckpt/11task/rd_controls.tex
```

`per_task_rd_analysis.py` reports, per task, ΔLast, ΔLearned (diagonal),
ΔForgetting, and ΔPre-exposure — the mean accuracy on a task over the stages
*before* it was trained, i.e. the per-task analogue of Transfer. With
`--zeroshot-csv` it also splits tasks at the median CLIP zero-shot accuracy and
gives Pearson/Spearman correlations between zero-shot accuracy and each delta,
which is the literal answer to "does it help overall or only on tasks close to
CLIP pretraining".

Both scripts print to stdout and write CSV plus a LaTeX table body.

## Reading the controls

- If **C1** matches ExRD, RD does not need the buffer and the buffer is only a
  label carrier; if Transfer drops, buffer contents do work that current-task
  images cannot.
- If **C2** holds Transfer up, the anchor's identity does not matter and any
  self-distillation signal suffices — which would weaken the paper's mechanistic
  claim, so report it either way. If Transfer instead falls toward the
  no-anchor ablation (64.38), the *frozen pre-trained* anchor is the
  load-bearing component, which is the claim the paper makes.
- **C3** vs ExRD isolates image identity with everything else fixed; C3 vs the
  "+ proportional replay" row (86.44 / 77.57 / 68.00) checks whether 0.3 extra
  ZSCL weight alone reproduces RD's Transfer gain.
