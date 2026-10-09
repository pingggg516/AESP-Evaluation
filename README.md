# AESP: Answer-Evidence Support Protocol

**Benchmarking answer-grounding consistency in remote sensing visual question answering.**

## Motivation

Visual grounding connects language to specific regions in an image. In grounded visual question answering, a vision-language model (VLM) provides these regions alongside its textual answer as visual evidence, allowing users to inspect which objects or image content support its response. This makes the relationship between the answer and the returned regions central to the purpose of grounding.

Yet this relationship can break down. A model may return a correct answer and accurately localize relevant objects, while the objects covered by its evidence do not collectively support what it says. The evidence may omit instances required by a stated count or fail to establish the relationship asserted in a comparison.

![Examples illustrating answer-grounding inconsistency in counting and comparison tasks.](figures/motivation.jpg)

*Figure 1. Correct answers and high localization scores do not guarantee that the returned grounding evidence supports the answer.*

For example, a model correctly answers that an image contains ten storage tanks but grounds only nine, achieving a grounding F1 score of 94.7%. Similarly, it correctly answers that tennis courts outnumber baseball fields but grounds two of each, achieving an F1 score of 88.9%. In both cases, the answer is correct, but the returned evidence fails to support its claim.

Standard answer-accuracy and localization metrics do not directly evaluate this relationship. Answer accuracy compares the response with the reference answer; localization metrics compare predicted regions with annotated object locations. High scores on these measurements can therefore coexist with a mismatch between the model's language answer and its visual evidence.

**This gap matters for the trustworthiness of grounded VLM responses.** Grounding gives users something concrete to inspect, but its value as evidence depends on whether it supports the accompanying claim. When this connection is missing, a response can appear visually justified without providing sufficient justification. Evaluating this connection is necessary to determine whether the returned regions deserve to be treated as supporting evidence.

We introduce **AESP (Answer-Evidence Support Protocol)** to explicitly assess this relationship in remote sensing VQA. AESP identifies annotated target objects covered by the predicted evidence regions and applies task-specific rules for existence, counting, and comparison to determine whether they support **the model's own answer**.

Together with answer accuracy and localization quality, evidence support provides a more informative assessment of grounded responses: whether the answer is correct, whether the regions are accurately localized, and whether their contents support the answer. It makes the evidence presented by a VLM auditable, while leaving the separate question of the model's internal reasoning process open.
