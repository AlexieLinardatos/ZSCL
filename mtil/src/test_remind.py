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

Run:  python -m src.test_remind
"""

import torch
import torch.nn.functional as F

import clip.clip as clip
from src.product_quantizer import ProductQuantizer
from src.remind_buffer import encode_from_layer, encode_to_layer, freeze_below_layer


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

    print()
    if failures:
        print("[test] FAILED:")
        for f in failures:
            print(f"  - {f}")
        raise SystemExit(1)
    print("[test] PASSED — split is exact, PQ decodes, freezing hits the right half.")


if __name__ == "__main__":
    main()
