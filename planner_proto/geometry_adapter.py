# geometry_adapter.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Sequence, Tuple
import json

from domain import Board, BoardMetadata, Component, Joint, save_board_json


# ===========================
# Налаштування / константи
# ===========================

# Мапа "сира" назва класу (YOLO/LS) → канонічне ім'я для планувальника.
CANONICAL_CLASS_MAP: Dict[str, str] = {
    # YOLO-клас для вертикальних електролітів
    "electrolyt_cap_vert": "ELECTROLYT_CAP_VERT",

    # Label Studio мок-класи (можеш доповнювати / міняти за потреби):
    "TO_263_5": "IC_TO263_5",
    "smd_diode": "SMD_DIODE",
    "smd_small_2pin": "SMD_SMALL_2PIN",
    "toroid_inductor_horz": "TOROID_INDUCTOR",
    "trim_potentiometer": "TRIM_POT",

    # запасний варіант — можна додавати будь-які інші сирі назви:
    # "THT_cyl_vert": "ELECTROLYT_CAP_VERT",
}

# За замовчуванням: скільки міліметрів відповідає одному пікселю.
# ЦЕ ТРЕБА ПІДКРУТИТИ ПІД РЕАЛЬНУ XL4015 (ширина плати / ширина зображення).
DEFAULT_PX_TO_MM: float = 0.0305614783226724

# Сторона плати за замовчуванням (поки що все на TOP).
DEFAULT_SIDE: str = "TOP"


# ===========================
# Допоміжні функції
# ===========================


def _load_raw_detections(path: Path) -> Dict[str, Any]:
    """
    Читає raw_detections.json (чи detections_xl.json) з форматом типу:

    {
      "image_path": "...",
      "image_size_px": [W, H],
      "detections": [
        {
          "id": 0,
          "class_id": 0,
          "class_name": "electrolyt_cap_vert",
          "conf": 0.73,
          "bbox_px": [x1, y1, x2, y2]
        },
        ...
      ]
    }
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    return data


def _merge_raw_detections(raw_list: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Об'єднує кілька raw-словників (наприклад, YOLO + LS) в один.
    Вважаємо, що всі вони про ту саму плату / зображення.
    """
    if not raw_list:
        raise ValueError("Порожній список raw detections для мерджу")

    base = raw_list[0]
    image_size_px = base.get("image_size_px") or base.get("image_size") or [0, 0]
    image_path = base.get("image_path")

    detections: List[Dict[str, Any]] = []
    for raw in raw_list:
        detections.extend(raw.get("detections", []))

    return {
        "image_path": image_path,
        "image_size_px": image_size_px,
        "detections": detections,
    }


def _bbox_px_to_mm(bbox_px: Sequence[float], px_to_mm: float) -> Tuple[float, float, float, float]:
    """Конвертує bbox [x1_px, y1_px, x2_px, y2_px] у мм."""
    x1_px, y1_px, x2_px, y2_px = bbox_px
    return (
        x1_px * px_to_mm,
        y1_px * px_to_mm,
        x2_px * px_to_mm,
        y2_px * px_to_mm,
    )


def _estimate_joints_mm(cls: str, bbox_mm: Tuple[float, float, float, float]) -> List[Joint]:
    """
    Грубо оцінює точки прогріву (joints) з bbox в мм.

    Тут свідомо простий "rule-based" хардкод:
      - ELECTROLYT_CAP_VERT: 2 точки по нижній стороні (дві площадки).
      - TOROID_INDUCTOR: одна точка в центрі.
      - решта: одна точка в центрі (мінімально достатньо для планувальника).
    """
    x1, y1, x2, y2 = bbox_mm
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    cls_upper = cls.upper()

    if "ELECTROLYT" in cls_upper:
        width = x2 - x1
        # Дві точки на нижній стороні прямокутника:
        x_left = x1 + 0.25 * width
        x_right = x1 + 0.75 * width
        y = y2
        return [Joint(x_mm=x_left, y_mm=y), Joint(x_mm=x_right, y_mm=y)]

    if "TOROID" in cls_upper:
        # Для тороїда поки що беремо одну точку в центрі.
        return [Joint(x_mm=cx, y_mm=cy)]

    # За замовчуванням — одна точка в центрі корпусу.
    return [Joint(x_mm=cx, y_mm=cy)]


def _canonical_class(raw_class_name: str) -> str:
    """
    Перетворює "сиру" назву класу (з YOLO/LS) у канонічну для планувальника.
    Якщо в мапі немає — повертає як є (але в upper() логіці вище
    все одно буде працювати).
    """
    return CANONICAL_CLASS_MAP.get(raw_class_name, raw_class_name)


# ===========================
# Основна логіка адаптера
# ===========================


def build_board_from_raw_dict(
    raw: Dict[str, Any],
    board_id: str,
    px_to_mm: float = DEFAULT_PX_TO_MM,
    default_side: str = DEFAULT_SIDE,
) -> Board:
    """
    Перетворює один merged raw-словник у Board (мм + компоненти).
    """
    image_path = raw.get("image_path")
    image_size_px = raw.get("image_size_px") or raw.get("image_size") or [0, 0]
    w_px, h_px = int(image_size_px[0]), int(image_size_px[1])

    meta = BoardMetadata(
        board_id=board_id,
        image_path=image_path,
        image_size_px=(w_px, h_px),
        px_to_mm=px_to_mm,
    )

    components: List[Component] = []
    per_class_counter: Dict[str, int] = {}

    for det in raw.get("detections", []):
        raw_cls = det.get("class_name")
        if raw_cls is None:
            # fallback, якщо немає class_name
            raw_cls = f"class_{det.get('class_id', 0)}"

        canon_cls = _canonical_class(raw_cls)

        bbox_px = det["bbox_px"]
        bbox_mm = _bbox_px_to_mm(bbox_px, px_to_mm)

        # Лічильник, щоб робити refdes типу ELECTROLYT_CAP_VERT_1, _2, ...
        per_class_counter[canon_cls] = per_class_counter.get(canon_cls, 0) + 1
        idx = per_class_counter[canon_cls]

        comp_id = det.get("id")
        if comp_id is None:
            comp_id = f"{canon_cls}_{idx}"

        refdes = f"{canon_cls}_{idx}"

        joints = _estimate_joints_mm(canon_cls, bbox_mm)

        comp = Component(
            id=str(comp_id),
            refdes=refdes,
            cls=canon_cls,
            side=default_side,
            bbox_mm=bbox_mm,
            joints_mm=joints,
        )
        components.append(comp)

    return Board(meta=meta, components=components)


def build_board_from_files(
    raw_paths: Sequence[Path | str],
    board_id: str,
    px_to_mm: float = DEFAULT_PX_TO_MM,
    default_side: str = DEFAULT_SIDE,
) -> Board:
    """
    Приймає один або кілька raw JSON-файлів (YOLO, LS),
    мерджить їх і повертає Board.
    """
    paths = [Path(p) for p in raw_paths]
    raw_list = [_load_raw_detections(p) for p in paths]
    merged_raw = _merge_raw_detections(raw_list)
    return build_board_from_raw_dict(
        merged_raw,
        board_id=board_id,
        px_to_mm=px_to_mm,
        default_side=default_side,
    )


def main() -> None:
    """
    Примірний CLI-виклик (можеш під себе змінити шляхи / board_id):

      python geometry_adapter.py

    Очікує:
      - reports/raw_detections.json  (або detections_xl.json – просто підправ шляхи)
    Дає:
      - reports/board.json  (канонічний формат для планувальника)
    """
    project_root = Path(__file__).resolve().parent
    raw_path = project_root / "reports" / "raw_detections.json"
    out_path = project_root / "reports" / "board.json"

    if not raw_path.is_file():
        raise FileNotFoundError(f"Не знайдено raw detections JSON: {raw_path}")

    board = build_board_from_files(
        [raw_path],
        board_id="XL4015_demo",
        px_to_mm=DEFAULT_PX_TO_MM,
        default_side=DEFAULT_SIDE,
    )

    save_board_json(out_path, board)
    print(f"Збережено canonical board JSON у {out_path}")


if __name__ == "__main__":
    main()
