
# Self-Supervised Learning with SimCLR on CIFAR-10 (Full Report)

This repository implements **SimCLR** (contrastive self-supervised learning) with a **ResNet-18** backbone on **CIFAR-10**, and evaluates learned representations using the two standard SSL protocols:

1) **Linear Evaluation (frozen encoder)** → measures *representation quality*  
2) **Fine-tuning (end-to-end)** → measures *downstream performance when allowed to learn*

> **Important:** GitHub image paths are **case-sensitive** (and Windows is not).  
> If an image shows locally but not on GitHub, it’s usually because the filename/casing or folder path in the README doesn’t match exactly.

---

## 1) What we trained (why there are 4 runs)

We ran **two experiments**, each with **two initializations**:

### A) Fine-tuning (10 epochs)
- **SimCLR-pretrained → fine-tune**
- **Scratch (random init) → fine-tune**

**What learns?** Encoder ✅ + classifier ✅  
**Why 10 epochs?** CIFAR-10 is small/easy, so end-to-end training converges quickly.

### B) Linear evaluation (30 epochs)
- **SimCLR-pretrained → linear eval**
- **Scratch → linear eval**

**What learns?** Encoder ❌ (frozen) + linear classifier ✅  
**Why 30 epochs?** Only one linear layer is trained, so convergence is slower → more epochs for stability.

---

## 2) Results (numbers)

### Linear Evaluation (Frozen Encoder, 30 epochs)
| Model | Best Accuracy |
|------|--------------:|
| Scratch (random encoder) | 40.89% |
| **SimCLR-pretrained encoder** | **58.87%** |

**Gain from SSL (representation quality):** **+17.98 percentage points**

### Fine-tuning (End-to-End, 10 epochs)
| Model | Best Accuracy |
|------|--------------:|
| Scratch | **89.18%** |
| SimCLR-pretrained | 89.08% |

**Interpretation:** With end-to-end supervision on CIFAR-10, a scratch model can catch up quickly.  
This does **not** contradict SSL. Linear eval is the “pure” test of representation quality.

---

## 3) Plots: Linear Evaluation (Frozen Encoder)

> Put the exported images here:
```
results/figures/linear_eval/
```

### Test Accuracy
![Linear Eval - Test Accuracy](results/figures/linear_eval/test_acc.png)

### Test Loss
![Linear Eval - Test Loss](results/figures/linear_eval/test_loss.png)

### Train Loss (epoch)
![Linear Eval - Train Loss (epoch)](results/figures/linear_eval/train_loss_epoch.png)

### Train Loss (step)
![Linear Eval - Train Loss (step)](results/figures/linear_eval/train_loss_step.png)

### Learning Rate
![Linear Eval - LR](results/figures/linear_eval/train_lr.png)

**What these plots show (why they matter):**
- Pretrained curve converges more smoothly and to a better accuracy
- Scratch has noisier optimization and worse final generalization
- Same LR schedule → fair comparison (difference comes from representation quality)

---

## 4) Plots: Fine-tuning (End-to-End)

> Put the exported fine-tuning images here:
```
results/figures/finetune/
```

### Test Accuracy
![Fine-tune - Test Accuracy](results/figures/finetune/test_acc.png)

### Test Loss
![Fine-tune - Test Loss](results/figures/finetune/test_loss.png)

### Train Loss (epoch)
![Fine-tune - Train Loss (epoch)](results/figures/finetune/train_loss_epoch.png)

### Train Loss (step)
![Fine-tune - Train Loss (step)](results/figures/finetune/train_loss_step.png)

### Learning Rate
![Fine-tune - LR](results/figures/finetune/train_lr.png)

**What these plots show:**
- Both runs converge similarly (CIFAR-10 is easy with full supervision)
- Fine-tuning can hide SSL advantages because the encoder is allowed to learn everything again

---

## 5) TensorBoard logs (where they are)

All runs were logged to TensorBoard under the `runs/` directory. You can compare everything together:

```bash
tensorboard --logdir runs
```

Or view by experiment type:

```bash
tensorboard --logdir runs/linear_eval
tensorboard --logdir runs/finetune
tensorboard --logdir runs/simclr_cifar10
```

---

## 6) How to reproduce

```bash
# SimCLR pretraining
python -m src.train_simclr

# Linear evaluation
python -m src.eval_linear --mode pretrained
python -m src.eval_linear --mode scratch

# Fine-tuning
python -m src.finetune --mode pretrained
python -m src.finetune --mode scratch
```

---

## 7) Recommended repo structure for plots

To avoid confusion between linear-eval and fine-tune screenshots, keep them separate and rename consistently:

```
results/figures/
├── linear_eval/
│   ├── test_acc.png
│   ├── test_loss.png
│   ├── train_loss_epoch.png
│   ├── train_loss_step.png
│   └── train_lr.png
└── finetune/
    ├── test_acc.png
    ├── test_loss.png
    ├── train_loss_epoch.png
    ├── train_loss_step.png
    └── train_lr.png
```

---

## 8) One-line takeaway (portfolio)

**SimCLR improves representation quality by ~18% in linear evaluation (58.87% vs 40.89%), demonstrating substantially more linearly separable features than random initialization.**
