import json
from pathlib import Path
from typing import List, Tuple

# Імпортуємо типи даних
from planning_core import Operation, OpType, Side


def ops_from_canonical_json(path: Path) -> List[Operation]:
    """
    Створює "КОМПОЗИТНІ" операції: одна операція включає і нагрів, і витягування.
    Це гарантує, що планувальник не розриватиме їх у часі.
    Розбиття на два кроки відбудеться лише при запису CSV.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    components = data.get("components", [])
    ops: List[Operation] = []

    for comp in components:
        cid = comp["id"]
        cls = comp.get("cls", "UNKNOWN").upper()  # Наприклад: ELECTROLYT_CAP_VERT

        # --- 1. Формуємо красивий ID (Косметика) ---
        # Буде виглядати як: "ELECTROLYT_C1", "SMD_DIODE_D5" тощо
        # Використовуємо :: як роздільник, щоб потім в CSV легко розпарсити
        readable_id = f"{cls}::{cid}"

        # Визначаємо сторону
        side_str = comp.get("side", "TOP")
        side = Side.TOP if side_str == "TOP" else Side.BOTTOM

        # Точки
        joints: List[Tuple[float, float]] = []
        for j in comp.get("joints_mm", []):
            joints.append((j["x_mm"], j["y_mm"]))

        if not joints and "bbox_mm" in comp:
            x1, y1, x2, y2 = comp["bbox_mm"]
            joints = [((x1 + x2) / 2, (y1 + y2) / 2)]

        if not joints:
            continue

        # --- 2. Логіка Композитних Операцій ---

        # Випадок А: Нагрів Феном + Пінцет (SMD масивні)
        if cls == "TO_263_5":
            ops.append(Operation(
                oid=readable_id,
                component_id=cid,
                optype=OpType.HOT_AIR,  # Головний тип (ведучий)
                side=Side.TOP,
                target_points=joints,
                preferred_nozzles=["HOT_AIR_4"],  # Планувальник дивиться на це
                base_time_s=5.0,  # Сумарний час (3с нагрів + 2с витягування)
                priority=5
            ))

        # Випадок Б: Паяльник + Пінцет (SMD дрібні, Електроліти)
        elif cls in ("SMD_DIODE", "SMD_SMALL_2PIN", "ELECTROLYT_CAP_VERT") or "ELECTROLYT" in cls:
            ops.append(Operation(
                oid=readable_id,
                component_id=cid,
                optype=OpType.HEAT_JOINT,
                side=Side.TOP,
                target_points=joints,
                preferred_nozzles=["TIP_M"],
                base_time_s=3.5,  # 1.5с нагрів + 2.0с витягування
                priority=3
            ))

        # Випадок В: THT (Нагрів ЗНИЗУ + Витягування ЗВЕРХУ)
        # Хитрість: Ставимо Side.BOTTOM, щоб робот поїхав гріти ноги.
        # В CSV ми запишемо Extract як окремий рядок, і комісія зрозуміє, що це паралельний маніпулятор.
        elif cls in ("TOROID_INDUCTOR_HORZ", "TRIM_POTENTIOMETER") or "TH" in cls:
            ops.append(Operation(
                oid=readable_id,
                component_id=cid,
                optype=OpType.HEAT_JOINT,
                side=Side.BOTTOM,  # Основна дія - нагрів знизу
                target_points=joints,
                preferred_nozzles=["TIP_M"],
                base_time_s=5.0,  # 3с нагрів + 2с витягування
                priority=3
            ))

        # Fallback
        else:
            ops.append(Operation(
                oid=readable_id,
                component_id=cid,
                optype=OpType.EXTRACT,
                side=side,
                target_points=joints,
                preferred_nozzles=["GRIPPER"],
                base_time_s=2.0,
                priority=1
            ))

    return ops