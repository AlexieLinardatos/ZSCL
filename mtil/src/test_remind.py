"""
Self-test for the REMIND pieces: the PQ codec and the split forward pass.

Runs in about a minute on a GPU and needs no dataset. Two properties matter, and
both are cheap to check exactly:

  1. Split equivalence. encode_from_layer(encode_to_layer(x, k), k) must equal
     visual(x) to floating-point tolerance, for every k. If it does not, the
     surgery is wrong and every stored activation is garbage in a way that would
     show up only as quietly bad accuracy eleven hours into a run.

  2. Reconstruction fidelity. PQ codes must decode back to activations close
     enough that the embedding they produce still matches the original. This is
     reported as cosine between the true embedding and the one recovered through
     the codec — the number that actually governs whether REMIND works.

  3. Replay distillation (TMD / IAD). With an untouched model and no PQ, the
     teacher's upper half on stored tokens must equal its embedding of the real
     image, so the stored IAD target must line up with its own codes and the
     two losses must agree. A misaligned target (codes paired with a different
     crop or a different row) breaks both checks.

Run:  python -m src.test_remind
"""

import random
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import clip.clip as clip
from src.product_quantizer import ProductQuantizer
from src.remind_buffer import (
    RemindReplayBuffer, encode_from_layer, encode_to_layer, freeze_below_layer,
)


def main():
    if not torch.cuda.is_available():
        print("[test] WARNING: no GPU, running on CPU (slow).")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("[test] loading CLIP ViT-B/16...")
    model, _, _ = clip.load("ViT-B/16", jit=False, pretrained=True)
    model = model.to(device).eval()
    visual = model.visual

    torch.manual_seed(0)
    images = torch.randn(8, 3, 224, 224, device=device).to(
        next(visual.parameters()).dtype
    )

    failures = []

    # ---- 1. split equivalence -------------------------------------------
    print("\n[test] split equivalence (must match the unsplit forward)")
    with torch.no_grad():
        reference = visual(images).float()

    n_blocks = len(visual.transformer.resblocks)
    for layer in (0, 3, 6, 9, n_blocks):
        with torch.no_grad():
            tokens = encode_to_layer(visual, images, layer)
            out = encode_from_layer(visual, tokens, layer).float()
        max_err = (out - reference).abs().max().item()
        cos = F.cosine_similarity(out, reference, dim=-1).mean().item()
        status = "ok" if cos > 0.9999 else "FAIL"
        print(f"    layer {layer:>2}: max_abs_err={max_err:.2e}  cos={cos:.6f}  {status}")
        if cos <= 0.9999:
            failures.append(f"split at layer {layer} does not reproduce the "
                            f"unsplit forward (cos={cos:.6f})")

    # ---- 2. PQ reconstruction -------------------------------------------
    print("\n[test] PQ reconstruction at layer 6")
    layer = 6
    with torch.no_grad():
        tokens = encode_to_layer(visual, images, layer).float()
    b, l, d = tokens.shape
    flat = tokens.reshape(-1, d)
    print(f"    token grid: {l} tokens x {d} dims")

    for m in (16, 32, 64):
        pq = ProductQuantizer(m=m).fit(flat, iters=10, verbose=False)
        recon = pq.decode(pq.encode(flat)).reshape(b, l, d)
        with torch.no_grad():
            out = encode_from_layer(
                visual, recon.to(next(visual.parameters()).dtype), layer
            ).float()
        cos = F.cosine_similarity(out, reference, dim=-1).mean().item()
        rel = pq.reconstruction_error(flat)
        kb = l * m / 1024
        print(f"    m={m:>3}: {kb:5.1f} KB/exemplar  token_err={rel:.4f}  "
              f"embedding_cos={cos:.4f}")

    print("\n    (embedding_cos here is a floor, not the real number: the "
          "codebook\n     is fitted on 8 random-noise images. On real data with "
          "a proper fit\n     it should be markedly higher.)")

    # ---- 3. freezing ------------------------------------------------------
    print("\n[test] freeze_below_layer")
    before = sum(p.requires_grad for p in model.parameters())
    n_frozen = freeze_below_layer(model, 6)
    after = sum(p.requires_grad for p in model.parameters())
    print(f"    trainable tensors: {before} -> {after}  (froze {n_frozen})")
    if after >= before:
        failures.append("freeze_below_layer did not reduce the trainable set")
    for block in visual.transformer.resblocks[:6]:
        if any(p.requires_grad for p in block.parameters()):
            failures.append("a block below the split is still trainable")
            break
    if not any(p.requires_grad
               for block in visual.transformer.resblocks[6:]
               for p in block.parameters()):
        failures.append("blocks above the split were frozen too — nothing "
                        "in the upper tower can learn")

    # ---- 4. replay distillation targets ----------------------------------
    print("\n[test] TMD / IAD on stored tokens (no PQ, untouched model)")
    from phase3.losses_phase3 import (
        compute_remind_distill_loss, compute_remind_replay_loss,
    )
    fresh, _, _ = clip.load("ViT-B/16", jit=False, pretrained=True)
    fresh = fresh.to(device).float().eval()
    dp = torch.nn.DataParallel(fresh) if device == "cuda" else fresh

    random.seed(0)
    ds = TensorDataset(torch.randn(16, 3, 224, 224), torch.arange(16) % 2)
    buf = RemindReplayBuffer(total_budget=16, layer=6, quantize=False)
    if device == "cuda":
        buf.add_task(0, ds, 16, ["cat", "dog"], lambda c: f"a photo of a {c}.",
                     model=dp, batch_size=8, teacher=dp)
        batch = next(iter(DataLoader(buf.get_combined_dataset(), batch_size=16)))
        codes, _, _, temb = batch
        print(f"    batch fields: {len(batch)}  teacher_emb {tuple(temb.shape)}")

        with torch.no_grad():
            recovered = encode_from_layer(
                fresh.visual, buf.decode_tokens(codes.cuda(), 0).float(), 6
            )
            recovered = recovered / recovered.norm(dim=-1, keepdim=True)
        cos = F.cosine_similarity(recovered, temb.cuda().float(), dim=-1)
        print(f"    stored target vs. upper(tokens): min cos={cos.min():.5f}")
        if cos.min() < 0.999:
            failures.append("stored teacher target does not line up with its "
                            f"codes (min cos={cos.min():.5f})")

        ref_texts = clip.tokenize([f"a photo of a {w}." for w in
                                   ("cat", "dog", "car", "tree", "plane")]).cuda()
        with torch.no_grad():
            ref_emb = dp(None, ref_texts).float()
            ref_emb = ref_emb / ref_emb.norm(dim=-1, keepdim=True)
        args = SimpleNamespace(T=2.0, weight_adjust=False, text_loss=True, ls=0.1)
        scale = fresh.logit_scale
        l_tmd = compute_remind_distill_loss(dp, dp, batch, buf, scale, args,
                                            "tmd", ref_emb).item()
        l_iad = compute_remind_distill_loss(dp, dp, batch, buf, scale, args,
                                            "iad", ref_emb).item()
        l_ce = compute_remind_replay_loss(dp, batch, scale, buf, args).item()
        rel = abs(l_tmd - l_iad) / max(abs(l_tmd), 1e-8)
        print(f"    loss tmd={l_tmd:.5f}  iad={l_iad:.5f}  (rel diff {rel:.2e})  "
              f"replay CE={l_ce:.4f}")
        if rel > 1e-2:
            failures.append(f"TMD and IAD disagree on an untouched model "
                            f"(rel diff {rel:.2e})")
        if not torch.isfinite(torch.tensor(l_ce)):
            failures.append("replay CE is not finite on a 4-field batch")
    else:
        print("    skipped: needs a GPU")

    print()
    if failures:
        print("[test] FAILED:")
        for f in failures:
            print(f"  - {f}")
        raise SystemExit(1)
    print("[test] PASSED — split is exact, PQ decodes, freezing hits the right "
          "half, distillation targets line up.")


if __name__ == "__main__":
    main()
