# yolo_proto/src/yolo_io.py
"""
yolo_io.py

1. Читає YOLO-txt анотації (експорт із Label Studio).
2. Перетворює їх у структуру, сумісну з detect_components.
3. Вміє малювати ці анотації на зображенні (використовуючи стиль YOLO).
"""

from __future__ import annotations

import json
import cv2  # Додаємо OpenCV для роботи з картинками
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from PIL import Image

# Імпортуємо малювальник з бібліотеки, яка в тебе вже встановлена
from ultralytics.utils.plotting import Annotator, colors

# ---------------------------------------------------------------------------
# Константи
# ---------------------------------------------------------------------------

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]  # .../pcb
DATASET_ROOT = PROJECT_ROOT / "yolo_proto" / "data" / "caps_xl_test"  # Шлях до твоїх тестів


# ---------------------------------------------------------------------------
# Внутрішні утиліти парсингу
# ---------------------------------------------------------------------------

def _load_class_names(classes_txt_path: Path) -> List[str]:
    if not classes_txt_path.is_file():
        # Якщо файла нема, повертаємо заглушки, щоб код не впав
        return [f"class_{i}" for i in range(100)]
    lines = classes_txt_path.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip()]


def _image_size(image_path: Path) -> tuple[int, int]:
    if not image_path.is_file():
        raise FileNotFoundError(f"Нема зображення: {image_path}")
    with Image.open(image_path) as img:
        return img.size


def _parse_yolo_label_file(
        label_path: Path,
        img_w: int,
        img_h: int,
        class_names: List[str],
        image_stem: str,
        next_det_id_start: int = 0,
) -> List[Dict[str, Any]]:
    detections: List[Dict[str, Any]] = []
    if not label_path.is_file():
        return detections

    lines = label_path.read_text(encoding="utf-8").splitlines()
    det_idx = next_det_id_start

    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5: continue

        try:
            cls_id = int(parts[0])
            x_c_n, y_c_n, w_n, h_n = map(float, parts[1:])
        except ValueError:
            continue

        # Безпечне отримання імені класу
        if 0 <= cls_id < len(class_names):
            cls_name = class_names[cls_id]
        else:
            cls_name = f"unknown_{cls_id}"

        # Денормалізація (з 0..1 у пікселі)
        x_c, y_c = x_c_n * img_w, y_c_n * img_h
        w, h = w_n * img_w, h_n * img_h
        x1, y1 = x_c - w / 2, y_c - h / 2
        x2, y2 = x_c + w / 2, y_c + h / 2

        det = {
            "id": f"{image_stem}-{det_idx}",
            "class_id": cls_id,
            "class_name": cls_name,
            "conf": 1.0,
            "bbox_px": [x1, y1, x2, y2],
        }
        detections.append(det)
        det_idx += 1

    return detections


# ---------------------------------------------------------------------------
# Публічний API
# ---------------------------------------------------------------------------

def get_single_sample(
        image_path: Path,
        labels_dir: Path,
        classes_txt: Path
) -> Dict[str, Any]:
    """Збирає дані для однієї картинки."""
    w, h = _image_size(image_path)
    label_path = labels_dir / f"{image_path.stem}.txt"
    class_names = _load_class_names(classes_txt)

    dets = _parse_yolo_label_file(label_path, w, h, class_names, image_path.stem)

    return {
        "image_path": str(image_path),
        "image_size_px": [w, h],
        "detections": dets
    }


# ... (весь код файлу вище - без змін)

def visualize_detections(
        sample: Dict[str, Any],
        save_path: Path,
        base_image_path: Optional[Path] = None
):
    """
    Малює bbox-и.
    :param sample: словник з даними (координати, класи).
    :param save_path: куди зберегти результат.
    :param base_image_path: (ОПЦІЙНО) якщо передано, малюємо поверх ЦІЄЇ картинки.
                            Якщо None - беремо оригінал з sample["image_path"].
    """
    # Визначаємо, на чому малюємо: на результаті YOLO чи на оригіналі
    if base_image_path is not None and Path(base_image_path).exists():
        src_path = Path(base_image_path)
    else:
        src_path = Path(sample["image_path"])

    if not src_path.exists():
        print(f"Помилка: не знайдено зображення-основу: {src_path}")
        return

    # Завантажуємо картинку
    im0 = cv2.imread(str(src_path))

    # Ініціалізуємо Анотатор
    # example=... потрібен для правильного авто-підбору шрифтів, беремо оригінал
    annotator = Annotator(im0, line_width=2, example=str(sample["image_path"]))

    for det in sample["detections"]:
        x1, y1, x2, y2 = det["bbox_px"]
        label = det["class_name"]

        # Малюємо
        annotator.box_label([x1, y1, x2, y2], label, color=colors(det["class_id"], True))

    # Зберігаємо
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), annotator.result())
    print(f"[yolo_io] Збережено комбінований результат: {save_path}")


# ... (кінець файлу CLI без змін)


# ---------------------------------------------------------------------------
# CLI для перевірки
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Налаштування для тесту xl_test_1
    # Припускаємо, що структура папок: caps_xl_test/ (там і картинка, і txt, і classes.txt)
    # Або txt лежить поруч з картинкою.

    # 1. Шляхи
    TARGET_IMG = DATASET_ROOT / "xl_test_1.jpg"
    # Якщо txt лежить поруч з картинкою (як часто буває при ручних тестах):
    LABELS_DIR = DATASET_ROOT
    CLASSES_TXT = DATASET_ROOT / "classes.txt"

    OUTPUT_VIS = PROJECT_ROOT / "runs" / "detect" / "manual_vis" / "xl_test_1_manual.jpg"

    print(f"--- Запуск yolo_io для {TARGET_IMG.name} ---")

    if TARGET_IMG.exists():
        # 2. Отримуємо дані (парсинг)
        sample = get_single_sample(TARGET_IMG, LABELS_DIR, CLASSES_TXT)
        print(f"Знайдено об'єктів: {len(sample['detections'])}")

        # 3. Малюємо (візуалізація)
        visualize_detections(sample, OUTPUT_VIS)

        # 4. (Опційно) Зберігаємо JSON
        json_out = OUTPUT_VIS.with_suffix(".json")
        json_out.write_text(json.dumps(sample, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"JSON записано: {json_out}")

    else:
        print(f"ПОМИЛКА: Не знайдено файл {TARGET_IMG}")
        print("Перевір шляхи в константах зверху файлу.")