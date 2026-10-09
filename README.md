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


## Benchmark

**GroundedRSVQA** contains **15,027 remote sensing images**, with one question per image and **163,399 relevant ground-truth object annotations**. Each sample includes a question, a reference answer, and the annotated objects needed to evaluate the answer and its visual evidence.

### Data composition

The benchmark is built from three object detection datasets:

| Source dataset | Images / QA pairs | Official source |
| --- | ---: | --- |
| DIOR | 11,714 | [Dataset page](https://gcheng-nwpu.github.io/#Datasets) |
| NWPU VHR-10 | 800 | [Dataset page](https://gcheng-nwpu.github.io/#Datasets) |
| SODA-A | 2,513 | [Project page](https://shaunyuan22.github.io/SODA/) |
| **Total** | **15,027** | |

It covers three question types:

| Task | Description | Samples | Answer format |
| --- | --- | ---: | --- |
| **Existence** | Whether a queried object class is present. | 8,592 | `"yes"` / `"no"` |
| **Counting** | The number of instances of a queried class. | 4,275 | Integer stored as a string |
| **Comparison** | Whether the first queried class has more instances than the second. | 2,160 | `"yes"` / `"no"` |

For comparison questions, equal counts correspond to `"no"`.

### Download

**Download GroundedRSVQA by clicking [Here](https://pan.baidu.com/s/1ltJ8L8xYvEkkoez1k7ZQ9Q?pwd=0516)**, and Extraction code: `0516`


The download contains:

```text
GroundedRSVQA/
├── images/                 # 15,027 images
├── annotations.jsonl       # One annotation record per image
└── README.md               # Data description, citations, and license information
```

Keep `images/` and `annotations.jsonl` in the same directory. Image paths in the annotation file are relative to this directory.

### Annotation format

`annotations.jsonl` contains one JSON object per line. The following actual sample is expanded for readability:

```json
{
  "sample_id": "00001",
  "image_id": "00001.jpg",
  "image_path": "images/00001.jpg",
  "image_width": 800,
  "image_height": 800,
  "task_type": "comparison",
  "question": "Are there more basketball courts than tennis courts in this aerial image?",
  "gt_answer": "no",
  "target_classes": ["basketball court", "tennis court"],
  "all_relevant_gt_objects": [
    {"class": "basketball court", "bbox": [386, 129, 728, 501]},
    {"class": "tennis court", "bbox": [242, 275, 533, 601]},
    {"class": "tennis court", "bbox": [78, 397, 370, 714]}
  ],
  "source_dataset": "DIOR",
  "source_image_id": "01800",
  "source_split": "trainval",
  "source_image_filename": "01800.jpg",
  "source_annotation_id": "01800"
}
```

| Field | Description |
| --- | --- |
| `sample_id` | Unique sample identifier, stored as a string. |
| `image_id`, `image_path` | Image filename and relative path. |
| `image_width`, `image_height` | Original image dimensions in pixels. |
| `task_type` | `existence`, `counting`, or `comparison`. |
| `question` | Question to present to the evaluated model. |
| `gt_answer` | Reference answer, stored as a string. |
| `target_classes` | Queried object classes, in question order. |
| `all_relevant_gt_objects` | All annotated instances of the queried classes, each with a class name and bounding box. |
| `source_dataset`, `source_split` | Original dataset and split. |
| `source_image_id`, `source_image_filename`, `source_annotation_id` | Identifiers linking the sample to its source image and annotation. |

**Bounding boxes use `[xmin, ymin, xmax, ymax]` in original-image pixels.** The first pair is the top-left corner; the second is the bottom-right corner.

`all_relevant_gt_objects` contains the complete set of annotated objects belonging to the queried classes. For a negative existence sample whose GT answer is `No`, this list is empty: `[]`.

These annotations provide the **ground truth** for evaluation. Users are responsible for running inference with the model they wish to evaluate on the benchmark images and questions. They then format the model outputs as required by AESP and submit that to AESP.

### Sources and citations

GroundedRSVQA adds question–answer pairs and organizes the source object annotations for grounded VQA. SODA-A polygon annotations are converted to horizontal bounding boxes using all polygon vertices.

Please acknowledge GroundedRSVQA/AESP and cite the original datasets when using this benchmark.

**DIOR**

Ke Li, Gang Wan, Gong Cheng, Liqiu Meng, and Junwei Han.  
[Object detection in optical remote sensing images: A survey and a new benchmark.](https://www.sciencedirect.com/science/article/pii/S0924271619302825)  
*ISPRS Journal of Photogrammetry and Remote Sensing*, 159:296–307, 2020.

**NWPU VHR-10**

- Gong Cheng, Junwei Han, Peicheng Zhou, and Lei Guo. [Multi-class geospatial object detection and geographic image classification based on collection of part detectors.](https://www.sciencedirect.com/science/article/abs/pii/S0924271614002524) *ISPRS Journal of Photogrammetry and Remote Sensing*, 98:119–132, 2014.
- Gong Cheng and Junwei Han. [A survey on object detection in optical remote sensing images.](https://www.sciencedirect.com/science/article/abs/pii/S0924271616300144) *ISPRS Journal of Photogrammetry and Remote Sensing*, 117:11–28, 2016.
- Gong Cheng, Peicheng Zhou, and Junwei Han. [Learning rotation-invariant convolutional neural networks for object detection in VHR optical remote sensing images.](https://ieeexplore.ieee.org/abstract/document/7560644) *IEEE Transactions on Geoscience and Remote Sensing*, 54(12):7405–7415, 2016.

**SODA-A**

Gong Cheng, Xiang Yuan, Xiwen Yao, Kebing Yan, Qinghua Zeng, Xingxing Xie, and Junwei Han.  
[Towards large-scale small object detection: Survey and benchmarks.](https://ieeexplore.ieee.org/document/10168277)  
*IEEE Transactions on Pattern Analysis and Machine Intelligence*, 45(11):13467–13488, 2023.

### License

The original datasets are provided under **[Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/)**, as stated on the [official dataset page](https://gcheng-nwpu.github.io/#Datasets) and the [SODA project page](https://shaunyuan22.github.io/SODA/#license). Their images and derived object annotations remain subject to these terms.

Noncommercial sharing and adaptation are permitted with appropriate attribution, a link to the license, and an indication of any changes. When redistributing the data, retain the source credits and license notices and do not impose restrictions beyond those permitted by the license.
