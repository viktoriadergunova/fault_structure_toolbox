---

## 2 . Installation

```bash
# clone repo
git clone <repo‑url> fault_structure_toolbox
cd fault_structure_toolbox

# create env & install deps
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 3 . Quick‑start notebook

```bash
jupyter notebook examples/demo.ipynb
```
Workflow example detection → post‑processing → evaluation.

---

## 4 . Folder layout

```
fault_structure_toolbox/          # project root 
│
├                         
│  
│
├── examples/
│   └── demo.ipynb                # demo.ipynb
│
├── toolbox/                      # package code
│   ├── __init__.py               # re‑exports functions
│   └── pipeline/
│       ├── __init__.py           # exports functions from pipeline(run, evaluate_mask, …)
│       ├── gabor.py              # detection 
│       ├── postprocessing.py     # skeleton → straight segments
│       └── evaluation.py         # rose + tile overlay
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 5 . API overview

| Helper | Source | Purpose |
|--------|--------|---------|
| `run` | `pipeline.gabor` | Apply Gabor bank, return `mask`, `orientation`, `magnitude`. |
| `debug_tiles` | `pipeline.gabor` | Show gray / mask / overlay for each tile. |
| `create_gabor_kernels` | `pipeline.gabor` | Build 16‑orientation kernel list. |
| `postprocess` | `pipeline.postprocessing` | Mask → skeleton → straight segments. |
| `evaluate_mask` | `pipeline.evaluation` | Rose diagrams + tile image overlay. |

Import from:

```python
from toolbox.pipeline import (
    run, debug_tiles, create_gabor_kernels,
    postprocess, evaluate_mask,
)
```

---

## 6 . Typical script

```python
from toolbox.pipeline import run, postprocess, evaluate_mask
import cv2

# 1 . detection
data = "IMG_PATH"
mask, orient, _ = run(data, crop=0) 

# 2 . load matching RGB (cropped)
rgb = cv2.imread(data)
rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

# 3 . fit straight segments
segments = postprocess(mask, rgb, tile_size=512)

# 4 . create PDF report
evaluate_mask(mask, orient, rgb=rgb, segments=segments,
              pdf_out="tile_roses_with_segments.pdf", tile_size=512)
```
---
