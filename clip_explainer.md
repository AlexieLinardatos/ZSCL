# CLIP explainer (session notes, 2026-10-07)

## 1. What CLIP is
Two encoders map into a shared 512-d space; both outputs are L2-normalized, so a dot product = cosine similarity.
**No activation layer at the end.** The "head" is: cosine similarity x logit_scale (learned temperature, ~100) -> softmax.

```
image -> Image Encoder (ViT-B/16) -> proj -> 512-d -> L2norm -> f_img
text  -> Text Encoder (Transformer) -> proj -> 512-d -> L2norm -> f_txt
```

## 2. Pretraining (where the shared space comes from)
Batch of N image-caption pairs -> N x N matrix S = scale * F_img @ F_txt.T.
Loss = symmetric cross-entropy (InfoNCE): softmax over rows (img->text) + over cols (text->img); target = diagonal.

```
                 T1     T2     T3   ...  TN
         I1   [I1.T1  I1.T2  I1.T3  ...      ]
         I2   [I2.T1  I2.T2  I2.T3  ...      ]
         ...                                  
         IN   [                     ... IN.TN]
   x logit_scale; loss = 1/2 CE(rows) + 1/2 CE(cols); target = diagonal
```

## 3. Zero-shot classification
1. **Build classifier from text (once per dataset):** for each class, fill prompt templates
   (`src/templates/`, e.g. "a photo of a {dog}."), encode (EOT token -> proj), normalize,
   average over templates, normalize again. Stack -> W [num_classes x 512].
2. **Classify:** preprocess (resize 224, center crop, CLIP mean/std) -> image encoder -> proj -> normalize -> f_img.
   `logits = logit_scale * f_img @ W.T` -> softmax -> argmax.
   Scale matters: cosines differ by ~0.01-0.1 between classes; x100 makes softmax sharp.

## 4. How this repo uses it (MTIL)
- **Fine-tuning:** CE on `scale * f_img @ W.T` for current-task classes.
  - `whole` mode: W recomputed by the live text encoder each step -> gradient into both encoders.
  - `fc` / `image-fc`: `ClassificationHead` = one `nn.Linear` (512 -> C), initialized from text embeddings, trained directly.
- **ZSCL:** ref images + ref sentences (Conceptual Captions) through frozen original CLIP and current model;
  match their similarity-matrix softmaxes so the space doesn't drift.
- **Feature replay (variant A):** store f_img (512-d fp16); recompute W with live text encoder.
  Replay CE = `scale * f_stored @ W.T`; only W carries gradient -> text encoder only.

```
buffer: f_img (stored, no grad) ---------------------+
                                                     v
old class names -> templates -> Text Encoder (live) -> W -> logits = scale * f . W^T -> CE
                                     ^                  |
                                     +--- gradient <----+   (text side only)
```

## 5. Why not freeze the head / text side instead?
- Nothing on the image side is frozen on purpose: stored features just can't backprop
  (the images/activations that made them are gone). The image encoder still trains via current-task CE, ZSCL, L_RD.
- In `whole` mode the "head" IS the text encoder's output, so freezing the head = freezing the text encoder.
  Then variant-A replay has zero gradient (f is constant, W is constant) -> does nothing.
- Freezing text doesn't stop forgetting: image encoder drifts away from fixed class embeddings.

| Setup | Replay updates | Problem |
|---|---|---|
| Pixel replay | image + text | 602 KB / sample |
| Freeze text, replay pixels | image only | text never adapts |
| Variant A | text only | stored features go stale as image encoder drifts (-> drift gate, adaptation ladder) |

## 6. Architecture sizes (ViT-B/16, ZSCL MTIL default; not checked against run configs)
| Part | Structure | Params |
|---|---|---|
| Image encoder | 16x16 patch conv -> 12 transformer blocks (width 768, 12 heads) -> LN -> proj 768->512 | ~86M |
| Text encoder | token+pos emb (77 ctx) -> 12 blocks (width 512, 8 heads, causal) -> LN -> proj 512->512 | ~63M |
| Classification head | 1 linear layer 512 -> C, no hidden layer/activation, weights = normalized text emb | 512*C |
| logit_scale | scalar | 1 |

Block = LN -> multi-head attention -> residual; LN -> MLP (4x width, GELU) -> residual.

## 7. The gradient math, from scratch
**Variables:** x = old image (not stored in A); theta_img / theta_txt = encoder weights;
f = ImageEncoder(x; theta_img) [512]; w_k = TextEncoder(prompt_k; theta_txt) [512]; W = stacked w_k;
s = logit_scale; z_k = s * (f . w_k); p = softmax(z); y = one-hot label; L = -log p_correct.

**Training:** theta <- theta - lr * dL/dtheta. Zero gradient = weight doesn't move.

**Chain rule:**
```
theta_img -> f --+
                 +-> z -> p -> L
theta_txt -> W --+

dL/dtheta_txt = dL/dz * dz/dW * dW/dtheta_txt      (nonzero: W recomputed live)
dL/dtheta_img = dL/dz * dz/df * df/dtheta_img      (zero in variant A: f loaded from disk, df/dtheta_img = 0)
```
Freeze text too -> neither side updates -> the loss value exists but changes nothing.

**Worked example** (2 classes, 2-d, s=1, no normalization):
```
f = [0.6, 0.8] (dog), w_dog = [1,0], w_cat = [0,1], y = [1,0]
z = [0.6, 0.8]; p = softmax(z) = [0.45, 0.55] (wrong); L = -log 0.45 = 0.80
dL/dz = p - y = [-0.55, +0.55]
dL/dw_dog = -0.55 * f = [-0.33, -0.44]  -> update moves w_dog TOWARD f
dL/dw_cat = +0.55 * f = [+0.33, +0.44]  -> update moves w_cat AWAY from f
dL/df = sum_k (p_k - y_k) w_k = [-0.55, 0.55] -> "move f toward w_dog", but f has no weights behind it -> goes nowhere
```
**Pixel replay:** store x, recompute f live -> df/dtheta_img != 0 -> both sides learn.
Variant A gives up the image-side gradient; fine because ZSCL + L_RD already supply it,
and replay is the only term training the text side for old classes.
