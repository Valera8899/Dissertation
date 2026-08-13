import json
from pathlib import Path

# .../pcb/yolo_proto/src/export_boxes.py
THIS_FILE = Path(__file__).resolve()
YOLO_ROOT = THIS_FILE.parents[1]      # .../pcb/yolo_proto
PROJECT_ROOT = THIS_FILE.parents[2]   # .../pcb

IN_JSON = PROJECT_ROOT / "reports" / "detections_demo.json"
OUT_JSON = PROJECT_ROOT / "reports" / "detections_mm.json"

# --- Параметри плати (їх можна буде підсунути ззовні/з cfg) ---

BOARD_WIDTH_MM = 120.0    # фізична ширина плати
BOARD_WIDTH_PX = 1600.0   # скільки пікселів по ширині відповідає цим 120 мм

MM_PER_PX = BOARD_WIDTH_MM / BOARD_WIDTH_PX


def convert_to_mm():
    if not IN_JSON.is_file():
        raise FileNotFoundError(f"Нема вхідного JSON з детекціями: {IN_JSON}")

    data = json.loads(IN_JSON.read_text(encoding="utf-8"))

    detections = data.get("detections", [])
    if not detections:
        print("У вхідному файлі немає детекцій.")
        return

    # якщо в JSON є розмір зображення — просто додаємо як метадані
    img_size = data.get("image_size_px", None)

    for det in detections:
        x1, y1, x2, y2 = det["bbox_px"]
        cx_px = (x1 + x2) / 2.0
        cy_px = (y1 + y2) / 2.0

        det["center_mm"] = [
            cx_px * MM_PER_PX,
            cy_px * MM_PER_PX,
        ]

    out_payload = {
        "image_path": data.get("image_path"),
        "image_size_px": img_size,
        "mm_per_px": MM_PER_PX,
        "board_width_mm": BOARD_WIDTH_MM,
        "detections": detections,
    }

    OUT_JSON.write_text(json.dumps(out_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Координати центрів у мм збережені в {OUT_JSON}")


if __name__ == "__main__":
    convert_to_mm()
