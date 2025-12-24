# planning_core.py
# Версія з реальним під’їздом до “баз” і станції інструментів.
# Є: пререквізити, батчинг за (side, nozzle), штраф за зміну насадки (setup),
# штраф/час за перехід між сторонами (side-switch), маршрут NN + 1-insertion.
# ПЕРЕД side-switch: доїзд до бази поточної сторони → flip → старт на базі нової сторони.
# ПЕРЕД setup насадки: доїзд до TOOL_STATION → setup.

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Set, Tuple as _Tuple
from enum import Enum, auto
import math

# --------- базові типи ---------

Point = Tuple[float, float]

class Side(Enum):
    TOP = auto()
    BOTTOM = auto()

class OpType(Enum):
    UNSCREW = auto()
    UNCLAMP = auto()
    HEAT_JOINT = auto()
    DESOLDER_SUCK = auto()
    HOT_AIR = auto()
    EXTRACT = auto()

class ToolModule(Enum):
    MECH_GRIPPER = auto()
    ELEC_SCREWDRIVER = auto()
    HOT_TIP = auto()
    HOT_AIR_NOZZLE = auto()
    DESOLDER_GUN = auto()

@dataclass
class Nozzle:
    module: 'ToolModule'
    name: str
    setup_time_s: float
    footprint_mm: float
    side_support: Tuple[Side, ...]  # на яких сторонах працює

@dataclass
class Operation:
    oid: str
    component_id: str
    optype: OpType
    side: Side
    target_points: List[Point]
    preferred_nozzles: List[str]
    base_time_s: float
    heat_time_s: float = 0.0
    priority: int = 0
    prerequisites: List[str] = field(default_factory=list)

@dataclass
class MachineConfig:
    travel_speed_mm_s: float = 200.0
    z_raise_time_s: float = 0.2
    settle_time_s: float = 0.1
    origin: Point = (0.0, 0.0)
    side_switch_time_s: float = 1.5

# --------- бази/станції ---------
BASE_TOP     = (0.0, 20.0) #площинні координати, на яких закріплена основа нагрівального двостороннього (№4)
BASE_BOTTOM  = (20.0, 0.0) #площинні координати, на яких закріплена основа механічного двостороннього (№3)
TOOL_STATION = (0.0, 0.0) #площинні координати станції інструментів

@dataclass
class PlanStep:
    op_id: str
    nozzle: str
    visit_point: Point
    travel_time_s: float
    op_time_s: float
    setup_time_s: float
    notes: str = ""

@dataclass
class PlanResult:
    steps: List[PlanStep]
    total_time_s: float

# --------- геометрія/метрики ---------

def dist(a: Point, b: Point) -> float:
    return math.hypot(a[0]-b[0], a[1]-b[1])

def centroid(points: List[Point]) -> Point:
    x = sum(p[0] for p in points)/len(points)
    y = sum(p[1] for p in points)/len(points)
    return (x, y)

def travel_time(a: Point, b: Point, cfg: MachineConfig) -> float:
    return dist(a, b) / cfg.travel_speed_mm_s

# --------- маршрут усередині батчу ---------

def nn_route(points: List[Point], start: Point) -> List[Point]:
    un = points.copy()
    curr = start
    order = []
    while un:
        nxt = min(un, key=lambda p: dist(curr, p))
        order.append(nxt)
        un.remove(nxt)
        curr = nxt
    return order

def path_len(route: List[Point]) -> float:
    return sum(dist(route[i], route[i+1]) for i in range(len(route)-1))

def one_insertion_improve(route: List[Point]) -> List[Point]:
    if len(route) < 3:
        return route
    best = route[:]
    best_len = path_len(best)
    improved = True
    while improved:
        improved = False
        for i in range(1, len(best)-1):
            for j in range(i+1, len(best)):
                r = best[:]
                node = r.pop(i)
                r.insert(j, node)
                L = path_len(r)
                if L < best_len:
                    best, best_len, improved = r, L, True
                    break
            if improved:
                break
    return best

# --------- батчинг та доступність ---------

def build_batches(ops: List[Operation]) -> Dict[Tuple[Side, str], List[Operation]]:
    """
    Ключ батчу: (SIDE, перша бажана насадка).
    Це мінімізує переходи між сторонами та кількість змін насадок.
    """
    batches: Dict[Tuple[Side, str], List[Operation]] = {}
    for op in ops:
        key = (op.side, op.preferred_nozzles[0] if op.preferred_nozzles else "MISC")
        batches.setdefault(key, []).append(op)
    for k in batches:
        batches[k].sort(key=lambda o: (o.priority, o.optype.name))
    return batches

def accessible(op: Operation, completed_ops: Set[str]) -> bool:
    # у цій версії доступність = лише виконані пререквізити
    return all(pr in completed_ops for pr in op.prerequisites)

def choose_nozzle(op: Operation, current_nozzle: Optional[str], nozzles: Dict[str, Nozzle]) -> Tuple[str, float]:
    best = None
    best_cost = float('inf')
    for nz_name in op.preferred_nozzles:
        nz = nozzles[nz_name]
        setup = 0.0 if current_nozzle == nz_name else nz.setup_time_s
        # енергії немає: вартість = setup + час дії
        cost = setup + op.base_time_s + 0.5 * op.heat_time_s
        if cost < best_cost:
            best_cost = cost
            best = nz_name
    return best, (0.0 if current_nozzle == best else nozzles[best].setup_time_s)

# --------- один прохід ---------

def plan_single_pass(
    ops: List[Operation],
    cfg: MachineConfig,
    nozzles: Dict[str, Nozzle],
    start_point: Point,
    current_nozzle: Optional[str],
    current_side: Optional[Side]
) -> _Tuple[PlanResult, Optional[str], Optional[Side], Point, List[Operation]]:
    completed: Set[str] = set()
    steps: List[PlanStep] = []
    total_time = 0.0
    remaining = ops[:]

    avail = [o for o in remaining if accessible(o, completed)]
    if not avail:
        return PlanResult([], 0.0), current_nozzle, current_side, start_point, remaining

    batches = build_batches(avail)
    curr_point = start_point

    # порядок батчів: спочатку поточна сторона (якщо є), потім інша
    def batch_order_key(key):
        side, nozzle = key
        side_pri = 0 if (current_side is None or side == current_side) else 1
        return (side_pri, nozzle)

    for (batch_side, nz_key) in sorted(batches.keys(), key=batch_order_key):
        group = batches[(batch_side, nz_key)]

        # --- Перехід між сторонами з під'їздом до бази поточної сторони ---
        if current_side is None or batch_side != current_side:
            if current_side is not None:
                base_from = BASE_TOP if current_side == Side.TOP else BASE_BOTTOM
                t_move_base = travel_time(curr_point, base_from, cfg)
                steps.append(PlanStep(
                    op_id=f"MOVE_TO_BASE_{current_side.name}",
                    nozzle=current_nozzle or "N/A",
                    visit_point=base_from,
                    travel_time_s=t_move_base,
                    op_time_s=0.0,
                    setup_time_s=0.0,
                    notes=f"Під’їзд до бази сторони {current_side.name} перед фліпом"
                ))
                total_time += t_move_base
                curr_point = base_from

            # сам side-switch
            steps.append(PlanStep(
                op_id=f"SETUP_SIDE_{batch_side.name}",
                nozzle=current_nozzle or "N/A",
                visit_point=curr_point,  # стоїмо на базі
                travel_time_s=0.0,
                op_time_s=0.0,
                setup_time_s=cfg.side_switch_time_s,
                notes=f"Перехід на сторону {batch_side.name}"
            ))
            total_time += cfg.side_switch_time_s

            # після фліпу стоїмо на базі нової сторони
            curr_point = BASE_TOP if batch_side == Side.TOP else BASE_BOTTOM
            current_side = batch_side

        # --- Вибір насадки ---
        dummy_op = group[0]
        chosen_nozzle, setup_t = choose_nozzle(dummy_op, current_nozzle, nozzles)

        # --- Під'їзд до станції інструментів та setup (раз на батч, якщо треба) ---
        if setup_t > 0:
            # доїзд до TOOL_STATION
            t_to_tool = travel_time(curr_point, TOOL_STATION, cfg)
            steps.append(PlanStep(
                op_id="MOVE_TO_TOOL_STATION",
                nozzle=current_nozzle or "N/A",
                visit_point=TOOL_STATION,
                travel_time_s=t_to_tool,
                op_time_s=0.0,
                setup_time_s=0.0,
                notes="Під’їзд до станції інструментів"
            ))
            total_time += t_to_tool
            curr_point = TOOL_STATION

            # сам setup насадки
            steps.append(PlanStep(
                op_id="SETUP_"+chosen_nozzle,
                nozzle=chosen_nozzle,
                visit_point=curr_point,  # на станції
                travel_time_s=0.0,
                op_time_s=0.0,
                setup_time_s=setup_t,
                notes="Переналаштування на насадку"
            ))
            total_time += setup_t
            current_nozzle = chosen_nozzle

        # --- Точки відвідування в батчі ---
        visit_points: List[Point] = []
        op_at_point: Dict[Point, List[Operation]] = {}
        for op in group:
            vp = op.target_points[0] if len(op.target_points) == 1 else centroid(op.target_points)
            visit_points.append(vp)
            op_at_point.setdefault(vp, []).append(op)

        # побудова маршруту усередині батчу:
        route = one_insertion_improve(nn_route(visit_points, curr_point))

        # --- Виконання батчу ---
        for vp in route:
            t_travel = travel_time(curr_point, vp, cfg)
            total_time += t_travel

            for op in op_at_point[vp]:
                t_op = cfg.z_raise_time_s + cfg.settle_time_s + op.base_time_s + op.heat_time_s
                steps.append(PlanStep(
                    op_id=op.oid, nozzle=current_nozzle or "N/A",
                    visit_point=vp,
                    travel_time_s=t_travel, op_time_s=t_op, setup_time_s=0.0,
                    notes=f"{op.optype.name} для компонента {op.component_id}"
                ))
                total_time += t_op
                completed.add(op.oid)

            curr_point = vp  # наступний старт — з цієї точки

    remaining = [o for o in remaining if o.oid not in completed]
    return PlanResult(steps, total_time), current_nozzle, current_side, curr_point, remaining

# --------- мульти-проходи ---------

def plan_full_job(
    all_ops: List[Operation],
    cfg: MachineConfig,
    nozzles: Dict[str, Nozzle],
    max_passes: int = 3
) -> PlanResult:
    start_point = cfg.origin
    curr_nozzle: Optional[str] = None
    curr_side: Optional[Side] = None
    grand_steps: List[PlanStep] = []
    total_time = 0.0
    remaining = all_ops[:]

    for p in range(1, max_passes+1):
        res, curr_nozzle, curr_side, start_point, remaining = plan_single_pass(
            remaining, cfg, nozzles, start_point, curr_nozzle, curr_side
        )
        grand_steps.extend(res.steps)
        total_time += res.total_time_s

        if not remaining:
            break

        # службовий крок RESCAN (позначка циклу)
        grand_steps.append(PlanStep(
            op_id=f"RESCAN_{p}",
            nozzle=curr_nozzle or "N/A",
            visit_point=start_point,
            travel_time_s=0.0, op_time_s=1.0, setup_time_s=0.0,
            notes="Проміжне сканування/оновлення"
        ))
        total_time += 1.0

    return PlanResult(grand_steps, total_time)
