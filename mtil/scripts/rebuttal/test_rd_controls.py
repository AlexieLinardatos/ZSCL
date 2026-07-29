"""
Stub-based smoke test for rd_controls.run_rd_control.

Checks the control wiring — which images and which teacher reach the RD loss,
batch-size matching, teacher caching, and the eval-time offload — without CUDA,
CLIP weights or even torch installed, by stubbing every dependency.  Run it
before submitting a 13.5 h control job:

    python mtil/scripts/rebuttal/test_rd_controls.py

It exercises the real patched functions; only the code they call is faked.
"""
import contextlib, os, sys, types
from argparse import Namespace

REBUTTAL = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REBUTTAL)

# ---------------- stub torch ----------------
torch = types.ModuleType("torch")
class FakeTensor:
    def __init__(self, name, n=64):
        self.name, self.shape = name, (n,)
    def __getitem__(self, s):
        return FakeTensor(f"{self.name}[:{s.stop}]", s.stop)
    def new_zeros(self, shape):
        return FakeTensor("zeros", 0)
    def __repr__(self):
        return f"<{self.name}>"
torch.cuda = types.SimpleNamespace(device_count=lambda: 1, empty_cache=lambda: None)
torch.nn = types.SimpleNamespace(DataParallel=lambda m, device_ids=None: m)
torch.no_grad = contextlib.nullcontext
torch.cat = lambda xs, dim=0: xs[0]
torch.load = lambda *a, **k: {}
sys.modules["torch"] = torch

# ---------------- stub clip / src / phase3 ----------------
clip_pkg = types.ModuleType("clip"); clip_mod = types.ModuleType("clip.clip")
clip_mod.load = lambda *a, **k: (object(), None, None)
clip_pkg.clip = clip_mod
sys.modules["clip"] = clip_pkg; sys.modules["clip.clip"] = clip_mod

src = types.ModuleType("src")
src_utils = types.ModuleType("src.utils")
src_utils.torch_load = lambda m, p: m
src_utils.seed_all = lambda s: None
src.utils = src_utils
sys.modules["src"] = src; sys.modules["src.utils"] = src_utils

calls = {}
def orig_get_next_batch(data_iter, dataset, args):
    return FakeTensor("current_images"), FakeTensor("labels"), data_iter
def orig_zscl(model, ref_model, ref_images, *a, **k):
    calls["zscl_images"] = ref_images
    return ("zscl_loss", "raw")
def orig_rd(model, ref_model, images, ref_texts, logit_scale, args, ref_embeddings=None):
    calls["rd"] = dict(teacher=ref_model, images=images, emb=ref_embeddings)
    return "rd_loss"
def orig_eval(*a, **k):
    calls["evaluated"] = True

tp3 = types.ModuleType("phase3.trainer_phase3")
tp3.get_next_batch = orig_get_next_batch
tp3.compute_zscl_loss = orig_zscl
tp3.compute_replay_teacher_distill_loss = orig_rd
tp3.evaluate_and_save = orig_eval
phase3 = types.ModuleType("phase3"); phase3.trainer_phase3 = tp3
args_p3 = types.ModuleType("phase3.args_phase3"); args_p3.parse_phase3_arguments = lambda: None
ft_p3 = types.ModuleType("phase3.finetune_phase3"); ft_p3.finetune_multi_task_phase3 = lambda a: None
sys.modules["phase3"] = phase3
sys.modules["phase3.trainer_phase3"] = tp3
sys.modules["phase3.args_phase3"] = args_p3
sys.modules["phase3.finetune_phase3"] = ft_p3

from rd_controls import run_rd_control as R

# ---------------- 1. patch installation ----------------
R._install_patches()
assert tp3.get_next_batch is R._patched_get_next_batch
assert tp3.compute_zscl_loss is R._patched_compute_zscl_loss
assert tp3.compute_replay_teacher_distill_loss is R._patched_rd_loss
assert tp3.evaluate_and_save is R._patched_evaluate_and_save
print("1. patches installed on all four trainer hooks: OK")

ARGS = Namespace(train_dataset="CIFAR100", save="/tmp/nonexistent_ckpt",
                 dataset_order=["Aircraft", "Caltech101", "CIFAR100"],
                 replay_batch_size=8, model="ViT-B/16")

def step(source, teacher="frozen"):
    """Simulate one trainer iteration in the order the trainer executes it."""
    calls.clear()
    R._S.image_source, R._S.teacher = source, teacher
    R._S.cur_images = R._S.ref_images = None
    tp3.get_next_batch(None, None, ARGS)                       # (1) CE batch
    tp3.compute_zscl_loss(None, "FROZEN", FakeTensor("ref_images"), None, None, ARGS)  # (3)
    tp3.compute_replay_teacher_distill_loss(                   # (5)
        None, "FROZEN", FakeTensor("replay_images", 8), FakeTensor("texts"),
        None, ARGS, ref_embeddings="FROZEN_EMB")
    return calls["rd"]

# ---------------- 2. image-source dispatch ----------------
assert step("replay")["images"].name == "replay_images"
assert step("current")["images"].name.startswith("current_images[:8]"), step("current")["images"]
assert step("reference")["images"].name.startswith("ref_images[:8]")
print("2. image sources dispatch to replay / current / reference: OK")

# ---------------- 3. batch-size matching ----------------
assert step("current")["images"].shape == (8,), "current batch must be capped at replay_batch_size"
R._S.match_batch = False
assert step("current")["images"].shape == (64,)
R._S.match_batch = True
print("3. RD batch capped at --replay_batch_size unless disabled: OK")

# ---------------- 4. frozen teacher is the default ----------------
r = step("replay")
assert r["teacher"] == "FROZEN" and r["emb"] == "FROZEN_EMB"
print("4. frozen teacher + its cached caption embeddings passed through: OK")

# ---------------- 5. prev-task teacher resolution & caching ----------------
built = []
def fake_build(args, ref_texts):
    built.append(args.train_dataset)
    return f"TEACHER({args.train_dataset})", f"EMB({args.train_dataset})"
R._build_prev_task_teacher = fake_build
r = step("replay", teacher="prev_task")
assert r["teacher"] == "TEACHER(CIFAR100)" and r["emb"] == "EMB(CIFAR100)", r
step("replay", teacher="prev_task")          # same task -> must reuse
assert built == ["CIFAR100"], built
ARGS.train_dataset = "Caltech101"            # next task -> must rebuild
step("replay", teacher="prev_task")
assert built == ["CIFAR100", "Caltech101"], built
print("5. prev-task teacher swapped in, cached per task, rebuilt on task change: OK")

# ---------------- 6. fallback when no previous checkpoint ----------------
R._build_prev_task_teacher = lambda args, ref_texts: (None, None)
R._S.teacher_task = None
r = step("replay", teacher="prev_task")
assert r["teacher"] == "FROZEN", r
print("6. falls back to frozen teacher when no previous checkpoint: OK")

# ---------------- 7. prev-task checkpoint path ----------------
R._build_prev_task_teacher = R.__dict__["_build_prev_task_teacher"]
a = Namespace(train_dataset="CIFAR100", save="/s", dataset_order=["Aircraft", "Caltech101", "CIFAR100"])
assert R._prev_task_ckpt_path(a) == "/s/Caltech101.pth"
a.train_dataset = "Aircraft"
assert R._prev_task_ckpt_path(a) is None      # first task has no predecessor
print("7. previous-task checkpoint path derived from dataset_order: OK")

# ---------------- 8. missing source images -> zero, never a crash ----------------
R._S.image_source = "current"; R._S.cur_images = None
out = tp3.compute_replay_teacher_distill_loss(
    None, "FROZEN", FakeTensor("replay_images", 8), None, None, ARGS)
assert isinstance(out, FakeTensor) and out.name == "zeros"
print("8. missing image source degrades to a zero loss: OK")

# ---------------- 9. eval offload wrapper ----------------
moved = []
class FakeTeacher:
    def cpu(self): moved.append("cpu")
    def cuda(self): moved.append("cuda")
R._S.teacher_model = FakeTeacher()
tp3.evaluate_and_save(None, None, None, 0)
assert moved == ["cpu", "cuda"] and calls.get("evaluated")
print("9. extra teacher offloaded to CPU during eval and restored: OK")

print("\nall checks passed")
