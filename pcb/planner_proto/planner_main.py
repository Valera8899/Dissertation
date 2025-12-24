# planner_main.py
# Легка версія з реалістичними заїздами до баз/станції.
# Є: батчинг за (side, nozzle), штраф setup і side-switch, summary з розкладом часу.
# На графіку видно і MOVE_*/SETUP_* кроки (фільтруємо тільки RESCAN_*).

import random
random.seed(42)

import json, csv
from pathlib import Path
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt

from planning_core import (
    Side, OpType, ToolModule, Nozzle, Operation, MachineConfig,
    plan_full_job, PlanResult,
    # маркери для графіка
    BASE_TOP, BASE_BOTTOM, TOOL_STATION
)

ROOT = Path(__file__).resolve().parents[1]          # .../pcb/
YOLO_OUT = ROOT / "yolo_proto" / "out" / "detections.json"
REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

# ---------- насадки ----------
NOZZLES: Dict[str, Nozzle] = {
    "TIP_S":       Nozzle(ToolModule.HOT_TIP, "TIP_S",       setup_time_s=8,  footprint_mm=1.0, side_support=(Side.BOTTOM,)),
    "TIP_M":       Nozzle(ToolModule.HOT_TIP, "TIP_M",       setup_time_s=10, footprint_mm=1.8, side_support=(Side.BOTTOM,)),
    "HOT_AIR_4":   Nozzle(ToolModule.HOT_AIR_NOZZLE, "HOT_AIR_4", setup_time_s=18, footprint_mm=4.0, side_support=(Side.TOP, Side.BOTTOM)),
    "DESOLDER":    Nozzle(ToolModule.DESOLDER_GUN, "DESOLDER",    setup_time_s=14, footprint_mm=1.5, side_support=(Side.BOTTOM,)),
    "GRIPPER":     Nozzle(ToolModule.MECH_GRIPPER, "GRIPPER",     setup_time_s=6,  footprint_mm=6.0, side_support=(Side.TOP, Side.BOTTOM)),
    "SCREWDRIVER": Nozzle(ToolModule.ELEC_SCREWDRIVER, "SCREWDRIVER", setup_time_s=7,  footprint_mm=5.0, side_support=(Side.TOP,))
}

# ---------- конвертер з YOLO ----------
def ops_from_yolo_json(path: Path) -> List[Operation]:
    """
    Очікуваний формат елементів:
    {
      "id": "IC_U5",
      "cls": "IC",
      "side": "TOP",
      "center": [80, 30],
      "joints": [[80,30]]
    }
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    items = data.get("items", [])
    ops: List[Operation] = []

    def side_of(s: str) -> Side:
        return Side.TOP if s.upper() == "TOP" else Side.BOTTOM

    for it in items:
        cid = it["id"]
        side = side_of(it.get("side", "TOP"))
        center: Tuple[float, float] = tuple(it["center"])
        joints: List[Tuple[float, float]] = [tuple(p) for p in it.get("joints", [center])]
        cls = it.get("cls", "").lower()

        if cls in ("electrolytic_vert", "tht_cyl_vert"):
            # Прогрів і відсмоктування знизу → витяг ЗІ ЗНИЗУ (корпус вниз)
            ops.append(Operation(f"J_{cid}", cid, OpType.HEAT_JOINT, Side.BOTTOM, joints, ["TIP_M"], base_time_s=1.5, heat_time_s=2.0, priority=3))
            ops.append(Operation(f"D_{cid}", cid, OpType.DESOLDER_SUCK, Side.BOTTOM, [center], ["DESOLDER"], base_time_s=2.0, priority=4, prerequisites=[f"J_{cid}"]))
            ops.append(Operation(f"E_{cid}", cid, OpType.EXTRACT, Side.BOTTOM, [center], ["GRIPPER"], base_time_s=1.8, priority=5, prerequisites=[f"D_{cid}"]))
        elif cls in ("ic", "qfp", "soic", "tqfp"):
            ops.append(Operation(f"H_{cid}", cid, OpType.HOT_AIR, Side.BOTTOM, [center], ["HOT_AIR_4"], base_time_s=3.0, heat_time_s=5.0, priority=3))
            ops.append(Operation(f"D_{cid}", cid, OpType.DESOLDER_SUCK, Side.BOTTOM, [center], ["DESOLDER"], base_time_s=2.0, priority=4, prerequisites=[f"H_{cid}"]))
            ops.append(Operation(f"E_{cid}", cid, OpType.EXTRACT, Side.TOP, [center], ["GRIPPER"], base_time_s=2.0, priority=5, prerequisites=[f"D_{cid}"]))
        elif cls in ("screw", "mount"):
            ops.append(Operation(f"U_{cid}", cid, OpType.UNSCREW, Side.TOP, [center], ["SCREWDRIVER"], base_time_s=2.0, priority=0))
        else:
            ops.append(Operation(f"E_{cid}", cid, OpType.EXTRACT, side, [center], ["GRIPPER"], base_time_s=1.5, priority=5))

    return ops

# ---------- демо-дані (без блокувань) ----------
def mock_ops() -> List[Operation]:
    ops: List[Operation] = [
        # Радіатор як набір дій
        Operation("U1", "HEATSINK_A", OpType.UNSCREW, Side.TOP, [(20, 40)], ["SCREWDRIVER"], base_time_s=2.0, priority=0),
        Operation("U2", "HEATSINK_A", OpType.UNSCREW, Side.TOP, [(35, 42)], ["SCREWDRIVER"], base_time_s=2.0, priority=0),
        Operation("U3", "HEATSINK_A", OpType.UNSCREW, Side.TOP, [(28, 55)], ["SCREWDRIVER"], base_time_s=2.0, priority=0),
        Operation("C1", "HEATSINK_A", OpType.UNCLAMP, Side.BOTTOM, [(27, 45)], ["GRIPPER"], base_time_s=1.5, priority=1, prerequisites=["U1","U2","U3"]),
        Operation("E1", "HEATSINK_A", OpType.EXTRACT, Side.TOP, [(27, 45)], ["GRIPPER"], base_time_s=2.5, priority=2, prerequisites=["C1"]),

        # IC
        Operation("H1", "IC_U5", OpType.HOT_AIR, Side.BOTTOM, [(80, 30)], ["HOT_AIR_4"], base_time_s=3.0, heat_time_s=5.0, priority=3),
        Operation("D1", "IC_U5", OpType.DESOLDER_SUCK, Side.BOTTOM, [(80, 30)], ["DESOLDER"], base_time_s=2.0, priority=4, prerequisites=["H1"]),
        Operation("E2", "IC_U5", OpType.EXTRACT, Side.TOP, [(80, 30)], ["GRIPPER"], base_time_s=2.0, priority=5, prerequisites=["D1"]),

        # THT-конденсатор: витяг ЗІ ЗНИЗУ
        Operation("J1", "C_ELEC1", OpType.HEAT_JOINT, Side.BOTTOM, [(120, 60), (123, 60)], ["TIP_M"], base_time_s=1.5, heat_time_s=2.0, priority=3),
        Operation("D2", "C_ELEC1", OpType.DESOLDER_SUCK, Side.BOTTOM, [(121.5, 60)], ["DESOLDER"], base_time_s=2.0, priority=4, prerequisites=["J1"]),
        Operation("E3", "C_ELEC1", OpType.EXTRACT, Side.BOTTOM, [(121.5, 60)], ["GRIPPER"], base_time_s=1.8, priority=5, prerequisites=["D2"]),
    ]
    return ops

# ---------- збереження артефактів ----------

def save_steps_csv(result: PlanResult, path_csv: Path) -> None:
    cum = 0.0
    with open(path_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["op_id","nozzle","x","y","travel_s","op_s","setup_s","cum_time_s","notes"])
        for s in result.steps:
            step_sum = s.travel_time_s + s.op_time_s + s.setup_time_s
            cum += step_sum
            x, y = s.visit_point
            w.writerow([s.op_id, s.nozzle, f"{x:.2f}", f"{y:.2f}",
                        f"{s.travel_time_s:.3f}", f"{s.op_time_s:.3f}",
                        f"{s.setup_time_s:.3f}", f"{cum:.3f}", s.notes])

def save_summary(result: PlanResult, path_txt: Path) -> None:
    with open(path_txt, "w", encoding="utf-8") as f:
        f.write(f"Кроків у плані: {len(result.steps)}\n")
        # розклад часу: виконання t_i, “штрафи” s_ij (setup + side-switch), переміщення
        total_ops = sum(s.op_time_s for s in result.steps)
        total_setups = sum(s.setup_time_s for s in result.steps)
        total_travel = sum(s.travel_time_s for s in result.steps)
        f.write(f"Час виконання операцій t_i (с): {total_ops:.2f}\n")
        f.write(f"Штрафи s_ij (setup + side-switch) (с): {total_setups:.2f}\n")
        f.write(f"Переміщення (с): {total_travel:.2f}\n")
        f.write(f"Сумарний час (с): {result.total_time_s:.2f}\n")

def save_route_png(result: PlanResult, path_png: Path) -> None:
    # Малюємо всі кроки, крім RESCAN_ — щоб на графіку були видимі MOVE_/SETUP_ заїзди
    pts = [(s.visit_point[0], s.visit_point[1]) for s in result.steps
           if not s.op_id.startswith("RESCAN_")]
    if not pts:
        return
    xs, ys = zip(*pts)
    plt.figure(figsize=(6, 5))
    plt.plot(xs, ys, marker="o")
    for i, (x, y) in enumerate(pts):
        plt.text(x, y, str(i+1))

    # маркери баз/станції
    bx, by = BASE_TOP;     plt.scatter([bx],[by]); plt.text(bx, by, "    B_TOP")
    bx, by = BASE_BOTTOM;  plt.scatter([bx],[by]); plt.text(bx, by, "   B_BOTTOM")
    sx, sy = TOOL_STATION; plt.scatter([sx],[sy]); plt.text(sx, sy, "    TOOL_S")

    plt.title("Маршрут із заїздами до баз/станції (setup & side-switch видимі)")
    plt.xlabel("X, мм"); plt.ylabel("Y, мм")
    plt.grid(True); plt.tight_layout(); plt.savefig(path_png, dpi=140); plt.close()

def dump_ops_csv(ops, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["oid","component_id","optype","side","x","y","preferred_nozzles","base_time_s","heat_time_s","priority","prerequisites"])
        for o in ops:
            x, y = (o.target_points[0] if len(o.target_points)==1
                    else (sum(p[0] for p in o.target_points)/len(o.target_points),
                          sum(p[1] for p in o.target_points)/len(o.target_points)))
            w.writerow([
                o.oid, o.component_id, o.optype.name, o.side.name,
                f"{x:.2f}", f"{y:.2f}", "|".join(o.preferred_nozzles),
                f"{o.base_time_s:.2f}", f"{o.heat_time_s:.2f}",
                o.priority, "|".join(o.prerequisites)
            ])

def dump_ops_json(ops, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = []
    for o in ops:
        payload.append({
            "oid": o.oid, "component_id": o.component_id,
            "optype": o.optype.name, "side": o.side.name,
            "target_points": o.target_points,
            "preferred_nozzles": o.preferred_nozzles,
            "base_time_s": o.base_time_s, "heat_time_s": o.heat_time_s,
            "priority": o.priority, "prerequisites": o.prerequisites
        })
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"items": payload}, f, ensure_ascii=False, indent=2)

# ---------- entrypoint ----------

def main():
    cfg = MachineConfig(travel_speed_mm_s=250.0, origin=(0.0, 0.0), side_switch_time_s=2.0)

    # джерело вхідних операцій
    if YOLO_OUT.exists():
        ops = ops_from_yolo_json(YOLO_OUT)
    else:
        ops = mock_ops()

    # дампимо “на чому рахували”
    dump_ops_csv(ops, REPORTS / "ops.csv")
    dump_ops_json(ops, REPORTS / "ops.json")

    # плануємо
    result = plan_full_job(ops, cfg, NOZZLES, max_passes=3)

    # консоль + файли
    print(f"Кроків у плані: {len(result.steps)}")
    print(f"Сумарний час (с): {result.total_time_s:.1f}")

    save_steps_csv(result, REPORTS / "steps.csv")
    save_summary(result, REPORTS / "summary.txt")
    save_route_png(result, REPORTS / "route.png")

if __name__ == "__main__":
    main()
