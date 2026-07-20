# Dead Channel

> *the sky was the color of television, tuned to a dead channel.*

![license](https://img.shields.io/badge/license-MIT-blue) ![data](https://img.shields.io/badge/raw%20per--pair%20scores-in%20results%2F-brightgreen) ![gpu](https://img.shields.io/badge/measured%20on-RTX%204090-76b900)

Measuring where diffusion quantization tunes to static. Post-quantization fidelity and latency for diffusion transformers, measured not quoted.

Every aggregate number in this README traces to raw per-pair JSON in [`results/`](results/). Mirror of the result data on Hugging Face at [felipesztutman/dead-channel-results](https://huggingface.co/datasets/felipesztutman/dead-channel-results).

Sibling of [kv-score](https://github.com/sztlink/kv-score), which does the same job for LLM KV-cache quantization. The field publishes speedups with a single self-reported fidelity table, usually one seed, usually no failure analysis. This repo measures fidelity the way it should be measured. Multi-seed, versioned prompts, paired against a full-precision reference, with the measurement conditions attached to every number.

## Study 1 . Krea 2 Turbo quant formats on RTX 4090 (2026-07-16)

[Krea 2](https://www.krea.ai/blog/krea-2-technical-report) is a 12.9B single-stream DiT with GQA attention, open weights since June 2026. Comfy-Org ships five quantized variants of the Turbo checkpoint. Nobody published a fidelity comparison between them. Here is one.

Setup. RTX 4090 24GB (Ada), ComfyUI master headless, torch 2.13 cu130, WSL2. 1024x1024, euler/simple, cfg 1.0. Reference is the BF16 checkpoint at 8 steps running with partial CPU offload. 12 prompts x 8 seeds gives n=96 pairs per variant, seed-locked to the reference. Prompts and seeds are versioned in [`harness/prompts.json`](harness/prompts.json) and never edited in place.

| variant | LPIPS (alex) | PSNR | ImageReward delta | s/img (batch) |
|---|---|---|---|---|
| **int8 convrot, 8 steps** | **0.066 +-0.055** | **27.7** | **-0.010** | ~4.3* |
| mxfp8, 8 steps | 0.115 +-0.080 | 24.0 | +0.004 | ~9.4* |
| fp8 scaled, 8 steps | 0.120 +-0.076 | 23.6 | -0.024 | 6.3 |
| fp8 scaled, 4 steps | 0.326 +-0.092 | 17.8 | -0.054 | 3.4 |
| fp8 scaled, 2 steps | 0.531 +-0.155 | 15.0 | **-0.523** | 2.3 |

*int8 and mxfp8 batch times ran under different loading conditions than the fp8 rows. Warm per-config latency needs a dedicated sweep before speed claims. Raw per-pair scores live in [`results/`](results/2026-07-16-krea2-turbo-rtx4090/).

### What the table says

**Rotation wins.** The int8 convrot variant, which rotates activations before quantizing in the QuaRot family tradition, is the most faithful quant of the five by a wide margin. Half the LPIPS of fp8 scaled, 4dB more PSNR, ImageReward delta indistinguishable from zero. On Ada hardware, where NVFP4 does not exist, rotation-based int8 currently beats both fp8 paths on fidelity.

**Steps are the expensive axis, not bits.** Dropping from 8 to 4 steps costs more fidelity than any quantization format here. Dropping to 2 steps collapses quality entirely, with ImageReward delta at -0.52. The Turbo checkpoint was distilled for 8 steps and it shows.

**A methodology note.** For step reduction, LPIPS measures trajectory divergence rather than quality loss. The 4-step image is a different composition, not a degraded one. ImageReward delta is the quality signal there, and at 4 steps it reads mild (-0.054) while LPIPS reads dramatic (0.326). Reading only one metric misleads in opposite directions depending on the axis. This distinction is absent from most published tables.

### Failure modes have signatures

Quantization hurts low-light scenes, fine texture and product shots. Step reduction hurts painterly styles hardest, with visible brushstroke structure the first casualty. The worst-case pairs per variant are listed in each results JSON.

![painterly, step collapse](samples/grid-painterly-step-collapse.png)
*Same seed, same prompt. BF16 ref, int8 convrot, mxfp8, then fp8 at 8, 4 and 2 steps. The painterly prompt survives every quant format at 8 steps and dies with step reduction.*

![low light, quant pain](samples/grid-lowlight-quant-pain.png)
*Low light is where quantization formats separate from each other.*

## Latency baseline (fp8 scaled, warm, same GPU)

| resolution | steps | s/img | DiT step |
|---|---|---|---|
| 1024 | 8 | 5.5 | 613ms |
| 1024 | 2 | 1.56 | |
| 768 | 8 | 2.96 | 329ms |
| 512 | 8 | 1.43 | 158ms |
| 512 | 2 | 0.42 | |

Step time scales near-linearly with token count at these resolutions (4x tokens costs 3.9x time), so the model is GEMM-bound rather than attention-bound here. ComfyUI's `--fast` fp8 matmul path adds only 13 to 15 percent on Ada. Most of the quantization speedup remains unharvested on this hardware, which is the motivation for the W4A4 work below.

## Study 2 . Ampere latency, RTX 3090 (2026-07-17)

Nobody had published quant-format latency for these checkpoints on Ampere. Here it is. Same protocol as the fidelity study, warm median of 3 seeds, ComfyUI on an RTX 3090 (sm86). Raw runs in [`results/2026-07-17-krea2-rtx3090-latency/`](results/2026-07-17-krea2-rtx3090-latency/).

Krea 2 Turbo, seconds per image at 1024x1024:

| variant | 8 steps | 4 steps | 2 steps |
|---|---|---|---|
| **int8 convrot** | **8.05** | **4.52** | **2.51** |
| fp8 scaled | 15.57 | 8.03 | 4.51 |
| mxfp8 | 17.76 | 9.03 | 5.02 |

The int8 convrot build is not only the most faithful (Study 1), it is also **roughly 2x faster than either fp8 path on Ampere**. On sm86 the INT8 tensor-core route is a real hardware path while the fp8 formats fall back to a slower one, so rotation-based int8 wins both axes at once here. This is the same variant that wins fidelity, which makes it the clear default on Ampere.

### The fused W4A4 path, measured

The tables above are all storage-quantized formats running through ComfyUI's general path. To see what a *fused* W4A4 kernel actually delivers, I benchmarked Nunchaku's INT4 FLUX.1-dev (SVDQuant) on the same RTX 3090. First published Ampere numbers for that runtime. Raw runs in [`results/2026-07-17-nunchaku-flux-rtx3090/`](results/2026-07-17-nunchaku-flux-rtx3090/).

| runtime + model | 1024px, per step |
|---|---|
| Nunchaku INT4 FLUX.1-dev (fused W4A4) | **~0.54s** |
| ComfyUI fp8 Krea 2 Turbo (storage fp8) | ~1.9s |

Both are 12B-class DiTs on the same card, so this is a cross-model indication rather than a controlled same-model A/B (a FLUX fp8/bf16 baseline on this card is still TODO). But the gap is stark: the fused W4A4 path runs roughly **3.5x faster per step** than the fp8-storage path. This is the empirical case for porting Krea 2 to a fused W4A4 runtime rather than settling for storage quantization, and [Study 3](#study-3--w4a4-krea-2-turbo-first-measured-2026-07-18) takes the first step by measuring the fidelity.

## Study 3 . W4A4 Krea 2 Turbo, built and measured (2026-07-18)

Every format in Study 1 is storage quantization. It shrinks the checkpoint and the memory, it does not touch the arithmetic of the forward pass. Study 2 showed, on a different model, that the *fused* W4A4 path runs about 3.5x faster per step because it rewrites that arithmetic. So the point was to take Krea 2 there. Nobody had quantized it to four-bit weights and four-bit activations, so the first job was to build the checkpoint, and the second was to measure it the way the field does not. This section is both. The build is the scarce part. The measurement is the part that tells you whether the build is worth running.

Setup. SVDQuant W4A4 through [deepcompressor](https://github.com/mit-han-lab/deepcompressor), calibrated on an H100 with 128 samples, smoothing plus a rank-32 low-rank branch that carries the outliers the 4-bit matmul cannot hold. The gated-attention output projection is quantized as a sibling of qkv. Guidance fixed at 0.0 because the Turbo checkpoint is cfg-distilled. Scored on the same RTX 4090 as Study 1, same 12 prompts x 8 seeds, seed-locked against the same BF16 reference. Raw per-pair scores in [`results/2026-07-18-krea2-w4a4/`](results/2026-07-18-krea2-w4a4/).

| variant | LPIPS (alex) | PSNR | ImageReward delta | forward pass |
|---|---|---|---|---|
| int8 convrot, 8 steps | 0.066 +-0.055 | 27.7 | -0.010 | storage |
| fp8 scaled, 8 steps | 0.120 +-0.076 | 23.6 | -0.024 | storage |
| **w4a4 svdquant, 8 steps** | **0.268 +-0.107** | **18.0** | **+0.052** | **fused** |
| fp8 scaled, 4 steps | 0.326 +-0.092 | 17.8 | -0.054 | storage |

### What the table says

**The reward model likes it more, and that is not the same as better.** The ImageReward delta of +0.052 is the highest of any variant measured here, storage or fused. I do not read that as four-bit beating full precision. A reward model is a learned preference with well-known failure modes, and the likelier reading is that quantization nudged the sampling trajectory into a more prototypical basin, closer to the kind of image the reward model was trained to reward. Reading a positive delta as a quality gain is exactly the trap this benchmark exists to avoid.

What the number does support is the narrower and more useful claim: **quality did not degrade.** Whatever four-bit costs, it does not show up as a penalty this reward model can see, which is the opposite of what step reduction does (-0.523 at 2 steps).

**The trajectory moves, and that is the whole cost.** LPIPS 0.268 lands near the fp8 4-step build, so the sampling path shifts to a neighboring composition. But unlike step reduction, which drags ImageReward down with it, W4A4 holds quality while it shifts. This is the signature of SVDQuant. The low-rank branch absorbs the activation outliers that grow block by block through the DiT (calibration error climbed monotonically from ~9700 at block 1 to ~28000 at block 27, and the branch soaked it up), so the predicted last-block collapse never appeared in the images.

**This is the only row that buys real time.** int8 convrot wins pure fidelity and stays the right default for storage quant on Ampere. But it runs through the general forward path. W4A4 is the only quantization here that rewrites the matmul into the fused low-bit kernel that turns into the ~3.5x of Study 2. The others make the file smaller. This one makes the machine faster.

### Failure modes

Fine texture is where W4A4 diverges most. The worst pairs are the `texture-detail` and `geometry` prompts (LPIPS up to 0.60), where the trajectory shift rewrites high-frequency structure. Portraits and painterly scenes hold closest to the reference. Full worst-case list in the results JSON.

## Study 4 . The W4A4 build runs in a fused runtime (2026-07-19)

Study 3 measured the fidelity of a W4A4 Krea 2 checkpoint through a fake-quantized forward pass. That answers whether the quality survives, not whether the build runs. So I ported Krea 2 into the [Nunchaku](https://github.com/nunchaku-tech/nunchaku) W4A4 runtime and ran it. This is the first time Krea 2 has run with four-bit weights and activations through fused low-bit kernels.

Both halves are open. The checkpoint is at [felipesztutman/Krea-2-Turbo-W4A4-Nunchaku](https://huggingface.co/felipesztutman/Krea-2-Turbo-W4A4-Nunchaku), the runtime port and the deepcompressor conversion path are at [sztlink/krea2-nunchaku](https://github.com/sztlink/krea2-nunchaku). No new CUDA was needed, the existing Nunchaku kernels already serve this shape, so the port is 213 lines of Python against them.

Setup. The 28 heavy transformer blocks are quantized (q/k/v/gate fused into one group, matching how the checkpoint was calibrated), everything else stays bf16. Measured on an L40S (Ada, the same generation as the 4090), 1024 and 512 px, guidance 0.0, warm timing, paired seed to seed against the bf16 model on the same card.

**It generates correct images.** Measured on the same 12 versioned prompts and 8 seeds as Studies 1 to 3, paired seed to seed against bf16 on the same card, 96 pairs per resolution.

| | LPIPS | PSNR | ImageReward delta | n |
|---|---|---|---|---|
| 1024px | 0.276 +-0.105 | 17.7 | +0.078 | 96 |
| 512px | 0.226 +-0.090 | 18.4 | -0.013 | 96 |

The perceptual distance at 1024px lands on Study 3's fake-quant prediction of 0.268, which is the strongest evidence that the fused runtime reproduces the calibration it was built from. ImageReward is positive at 1024 and flat at 512, so quality did not degrade, it moved a touch the other way. At 512px, the resolution a streaming pipeline would actually run, fidelity is better than at 1024. Per prompt the pattern is the one this benchmark keeps finding, hands and dense scenes and low light diverge most (0.35 to 0.37), portraits and product shots hold closest (0.14 to 0.20). The high-LPIPS prompts shift their composition, they do not lose quality.

![bf16 vs W4A4 fox](samples/foxpair-bf16-w4a4.png)
*Same seed, same prompt. bf16 left, four-bit W4A4 right. LPIPS 0.14.*

![bf16 vs W4A4 across five prompts](samples/grid-w4a4-fused-runtime.png)
*bf16 left, W4A4 right, five prompts. Quality holds. The prompts that move most (texture, dense cityscape) shift their composition, they do not degrade.*

Latency, warm median of 3 seeds, both models running the same expanded-head attention path so the difference is the four-bit arithmetic alone:

| resolution, steps | W4A4 | bf16 | speedup |
|---|---|---|---|
| 1024px, 8 steps | 5.36s | 7.71s | 1.44x |
| 512px, 8 steps | 1.34s | 2.19s | 1.63x |
| 512px, 4 steps | 0.73s | n/a | 1.38 img/s |
| 512px, 2 steps | 0.41s | n/a | 2.45 img/s |

Both columns run FlashAttention, so the speedup is the four-bit arithmetic alone, 1.44x at 1024 and 1.63x at 512. It grows as resolution drops because attention shrinks and the quantized matmuls take a larger share. A separate 3x sits on top of both columns and is described below, which is why getting the attention path right mattered more than the quantization did.

### The grouped-query flag costs 3x when a mask is present

The first port ran at 20 seconds for eight steps and a profile put six of every eight seconds inside attention, in the math backend. The cause is a one-line trap. Krea 2 uses grouped-query attention, 48 query heads against 12 key/value heads, and the natural way to express that is `enable_gqa=True` on PyTorch's scaled dot product attention. Krea 2 also always hands attention a mask, because text and image tokens share one sequence. Flash does not serve those two together. Pass both and SDPA drops to the math backend, materializes the full attention matrix, and nothing warns you. Expanding the 12 key/value heads up to 48 by hand and calling plain multi-head attention lets FlashAttention engage with the mask still in place.

A four-way ablation on the real pipeline, 1024px and 8 steps, isolates it:

| | plain | with dtype cast |
|---|---|---|
| enable_gqa=True | 22.3s | 22.4s |
| expand k/v heads | 7.7s | 7.8s |

The expansion is the whole effect, about 3x. A dtype cast I had suspected makes no measurable difference. I want to flag how I got this wrong first, because it is the useful part. I originally blamed the flag, then ran an ablation on scaled dot product attention in isolation, saw `enable_gqa` reach the flash backend fine, and corrected myself to a dtype story. That isolated test had no attention mask, so it did not reproduce the condition that matters. Only the ablation inside the real pipeline, with the mask the model actually passes, settles it. The lesson is not about this model. Any GQA transformer that also masks will silently lose FlashAttention through that flag.

### Where the rest of the speedup lives

This runtime keeps normalization, rotary and the attention itself eager and only the matmuls are fused, so the four-bit win is partly masked by per-projection activation quantization. Folding the quantize, the matmul, the q/k norm and rotary into a single kernel (the pattern Nunchaku already uses for other models) is the next lever. For Krea 2 it needs the sigmoid gate handled inside that fused path, which is the kernel-level work this study sets up.

## Run it

```bash
# score a variant directory against a reference directory
# images paired by filename: {prompt_id}__{seed}.png
python harness/dit_score.py \
  --reference out/ref-bf16-1024-8s \
  --variant   out/int8convrot-1024-8s \
  --prompts   harness/prompts.json \
  --out       scores/int8convrot.json
```

Requires torch, lpips, image-reward and a pinned transformers 4.49 (ImageReward's BLIP breaks on newer). Generation scripts for the ComfyUI API are in `harness/`.

## Caveats

- Single GPU, single day, one model family. Ampere numbers (RTX 3090) are next.
- The BF16 reference itself runs with partial offload on 24GB. Bit-identical BF16 on larger VRAM was not verified against it.
- Batch s/img figures include VAE decode, text encode cache hits and HTTP overhead. Only the fp8 rows come from a controlled warm sweep.
- ImageReward is one reward model with known style biases. It anchors the quality axis here, it does not close the question.

## Roadmap

- **Port the measured W4A4 checkpoint into a fused runtime.** Study 3 proves the fidelity holds. Next is running it through [Nunchaku](https://github.com/nunchaku-ai/nunchaku) on the 3090/4090 to convert the ~3.5x of Study 2 into real frames, with this harness as the gate at every step.
- **Variable-bit W4A4, allocated per block from measured need.** Study 3 quantizes every block to the same 4 bits. The calibration error climbs ~2.9x from first block to last, which says the bit budget is misallocated. A data-driven per-block allocation, spending bits where the DiT actually needs them, is the obvious next lever. This borrows directly from [@spiritbuun's VBR](https://x.com/spiritbuun) format for LLM KV caches, which does exactly this layer by layer, and from the depth-resolved harm profiling in [kv-score](https://github.com/sztlink/kv-score).
- Ampere (RTX 3090) numbers for every storage variant. No official benchmarks exist for that hardware.
- Composition of W4A4 with temporal feature caching for streaming img2img.

## Who

I build low-level inference for image models and installations that run on them. More at [felipesztutman.com/lab](https://felipesztutman.com/lab).
