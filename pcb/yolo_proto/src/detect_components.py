"""
yolo_proto/src/detect_components.py

Шар YOLO-детекції:
- ганяє натреновану модель по одному зображенню (боєвий режим)
  або по директорії (дебаг);
- віддає "raw detections" у пікселях у узгодженому форматі,
  який далі їсть geometry_adapter.

!!! ВАЖЛИВО:
- НІЧОГО не знає про planning_core / operations / маршрути.
- Тільки YOLO → bbox_px у пікселях.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import json

import torch
from ultralytics import YOLO


# ----------------- Шляхи всередині проєкту -----------------


THIS_FILE = Path(__file__).resolve()
YOLO_ROOT = THIS_FILE.parents[1]      # .../pcb/yolo_proto
PROJECT_ROOT = THIS_FILE.parents[2]   # .../pcb

# За замовчуванням беремо останній твій нормальний ран train4
DEFAULT_MODEL_PATH = PROJECT_ROOT / "runs" / "detect" / "train4" / "weights" / "best.pt"

# Куди за замовчуванням складати JSON-репорти
REPORTS_DIR = PROJECT_ROOT / "reports"


# ----------------- Внутрішні хелпери -----------------


def _ensure_model(model_path: Union[str, Path]) -> YOLO:
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Нема ваг моделі: {model_path}")

    # щоб матриці не душили CPU/GPU
    torch.set_float32_matmul_precision("high")

    return YOLO(str(model_path))


def _infer_single_image(
    image_path: Union[str, Path],
    model: YOLO,
    conf: float = 0.05,
    imgsz: int = 1024,
    save_vis: bool = False,
    vis_subdir: str = "xl_infer",
) -> Dict[str, Any]:
    """
    Проганяє одну картинку через модель і повертає словник формату:

    {
      "image_path": "yolo_proto/data/caps_xl_test/xl_test_2.jpg",
      "image_size_px": [W, H],
      "detections": [
        {
          "id": "C1",
          "class_id": 0,
          "class_name": "electrolyt_cap_vert",
          "conf": 0.93,
          "bbox_px": [x1, y1, x2, y2]
        },
        ...
      ]
    }
    """
    image_path = Path(image_path)

    if not image_path.is_file():
        raise FileNotFoundError(f"Нема зображення: {image_path}")

    # Спробуємо відносний шлях від кореня проекту — зручніше в JSON
    try:
        rel_image_path = str(image_path.relative_to(PROJECT_ROOT))
    except ValueError:
        rel_image_path = str(image_path)

    # Запуск інференсу
    results = model(
        source=str(image_path),
        imgsz=imgsz,
        conf=conf,
        device="cpu",  # якщо захочеш — міняєш на "0" для GPU
        save=save_vis,
        project=str(PROJECT_ROOT / "runs" / "detect"),
        name=vis_subdir,
        exist_ok=True,
        verbose=False,
    )

    res = results[0]
    h, w = res.orig_shape  # (height, width)

    detections: List[Dict[str, Any]] = []

    # res.boxes.xyxy вже в пікселях
    names_map = model.names  # dict: class_id -> class_name

    for i, box in enumerate(res.boxes):
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf_score = float(box.conf[0])
        cls_id = int(box.cls[0])
        cls_name = names_map.get(cls_id, f"class_{cls_id}")

        det = {
            "id": f"C{i+1}",  # простий ID всередині кадру
            "class_id": cls_id,
            "class_name": cls_name,
            "conf": round(conf_score, 4),
            "bbox_px": [x1, y1, x2, y2],
        }
        detections.append(det)

    payload: Dict[str, Any] = {
        "image_path": rel_image_path,
        "image_size_px": [w, h],
        "detections": detections,
    }
    return payload


def _write_json(obj: Any, out_path: Union[str, Path]) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


# ----------------- Публічне API -----------------


def run_single_image(
    image_path: Union[str, Path],
    model_path: Optional[Union[str, Path]] = None,
    conf: float = 0.05,
    imgsz: int = 1024,
    out_json_path: Optional[Union[str, Path]] = None,
    save_vis: bool = True,
) -> Dict[str, Any]:
    """
    Основна точка входу, яку ти вже викликаєш із planner_main.

    Параметри:
    - image_path: шлях до зображення (Path або str).
    - model_path: шлях до best.pt (якщо None — DEFAULT_MODEL_PATH).
    - conf: поріг впевненості.
    - imgsz: розмір інференсу YOLO.
    - out_json_path: якщо задано — записує "raw detections" у JSON.
    - save_vis: чи зберігати промальовані зображення в runs/detect/xl_infer.

    Повертає:
    - payload dict у форматі "raw detections" (див. _infer_single_image).
    """
    if model_path is None:
        model_path = DEFAULT_MODEL_PATH

    model = _ensure_model(model_path)

    payload = _infer_single_image(
        image_path=image_path,
        model=model,
        conf=conf,
        imgsz=imgsz,
        save_vis=save_vis,
        vis_subdir="xl_infer",
    )

    if out_json_path is not None:
        _write_json(payload, out_json_path)

    return payload


def run_on_directory(
    images_dir: Union[str, Path],
    model_path: Optional[Union[str, Path]] = None,
    conf: float = 0.05,
    imgsz: int = 1024,
    out_json_path: Optional[Union[str, Path]] = None,
    save_vis: bool = True,
) -> List[Dict[str, Any]]:
    """
    Дебажний режим: проганяє ВСІ картинки з директорії
    (*.jpg, *.jpeg, *.png) і повертає список payload-ів.

    Якщо задано out_json_path, записує список у JSON
    (так ти робив з detections_xl.json).
    """
    images_dir = Path(images_dir)
    if not images_dir.is_dir():
        raise NotADirectoryError(f"Нема директорії з зображеннями: {images_dir}")

    if model_path is None:
        model_path = DEFAULT_MODEL_PATH

    model = _ensure_model(model_path)

    image_paths: List[Path] = []
    for pattern in ("*.jpg", "*.jpeg", "*.png"):
        image_paths.extend(sorted(images_dir.glob(pattern)))

    if not image_paths:
        raise FileNotFoundError(f"У {images_dir} не знайдено жодного JPG/PNG")

    payloads: List[Dict[str, Any]] = []
    for img_path in image_paths:
        payload = _infer_single_image(
            image_path=img_path,
            model=model,
            conf=conf,
            imgsz=imgsz,
            save_vis=save_vis,
            vis_subdir="xl_infer",
        )
        payloads.append(payload)

    if out_json_path is not None:
        _write_json(payloads, out_json_path)

    return payloads


# ----------------- CLI / локальний дебаг -----------------


if __name__ == "__main__":
    # Проста ручна перевірка, щоб не лазити в planner_main під час дебагу.
    test_dir = YOLO_ROOT / "data" / "caps_xl_test"
    OUT_PATH = REPORTS_DIR / "detections_xl.json"

    print(f"[detect_components] Запускаю інференс по директорії: {test_dir}")
    payloads = run_on_directory(
        images_dir=test_dir,
        model_path=DEFAULT_MODEL_PATH,
        conf=0.05,
        imgsz=1024,
        out_json_path=OUT_PATH,
        save_vis=True,
    )
    print(f"[detect_components] Оброблено {len(payloads)} зображень, JSON: {OUT_PATH}")
