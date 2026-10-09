# AESP: Answer-Evidence Support Protocol

**Benchmarking answer-grounding consistency in remote sensing visual question answering.**

## Motivation

### Grounding as visual evidence

Visual grounding connects language to specific regions in an image. In grounded visual question answering, a vision-language model (VLM) returns these regions alongside its answer as visual evidence. Users can then inspect the objects presented as support for the model's response.

For grounding to serve this purpose, **the returned evidence should support what the model says.**

### When the evidence does not support the answer

A model can answer correctly and localize relevant objects accurately, yet return evidence that fails to support its answer. Our counting and comparison examples illustrate this problem.

![Answer-grounding inconsistency in counting and comparison tasks.](figures/motivation.jpg)

*Figure 1. Correct answers and high localization scores do not guarantee answer-grounding consistency.*

| Task | Model's answer | Returned evidence | Grounding F1 |
| --- | --- | --- | --- |
| Counting | There are **10 storage tanks**. | Only **9 storage tanks** are covered. | **94.7%** |
| Comparison | There are **more tennis courts than baseball fields**. | **2 tennis courts and 2 baseball fields** are covered. | **88.9%** |

Both answers are correct. However, nine grounded tanks do not establish a count of ten, and equal grounded counts do not establish “more than.”

> **Relevant objects do not necessarily provide sufficient evidence for the answer.**

### What standard metrics miss

Answer accuracy and localization quality assess different parts of a grounded response. Neither directly checks the logical relationship between them.

| Evaluation dimension | Question being evaluated |
| --- | --- |
| **Answer accuracy** | Does the model's answer match the reference answer? |
| **Localization quality** | Do the predicted regions match the annotated object locations? |
| **Evidence support — AESP** | Do the objects covered by the predicted regions support the model's own answer? |

The third question cannot be answered simply by reporting high scores on the first two.

### From grounding to trustworthy answers

This gap affects the trustworthiness of grounded VLM responses. Users expect the returned regions to provide evidence for the answer. If the objects in those regions do not support what the model says, the grounding cannot serve as reliable evidence.

We introduce **AESP (Answer-Evidence Support Protocol)** to explicitly evaluate this connection in remote sensing VQA. AESP identifies annotated target objects covered by the predicted evidence regions and checks whether they support the model's answer under task-specific rules:

- **Existence:** whether the covered objects support the stated presence or absence.
- **Counting:** whether the covered objects support the stated count.
- **Comparison:** whether the covered objects support the stated numerical relationship.

Evaluated alongside answer accuracy and localization quality, AESP distinguishes correct answers with supporting evidence from correct answers with unsupported claims and assesses the evidence that the model presents to users.
