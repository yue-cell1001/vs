import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from dataclasses import asdict, dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from threading import Lock
from decimal import Decimal
from pathlib import Path
from time import perf_counter

from common.data import load_network
from common.multiobjective import hypervolume, layers2d
from common.pareto import objective_key
from stage1.params import Params as Stage1Params
from stage1.constraints import violations
from stage2.params import Params as Stage2Params
from stage2.bounds import objective_bounds
from stage2.audit import audit_file
from stage2.scalarization import reference_scales

PROJECT = Path(__file__).resolve().parent
CAPACITIES = (4, 5, 6, 7, 8)


@dataclass(frozen=True)
class Settings:
    capacities: tuple = CAPACITIES
    iterations: int = 20
    population: int = Stage1Params().population
    generations: int = Stage1Params().generations
    evaluation_budget: int = 0
    seed: int = 2026
    subproblem_seconds: float | None = None

    def __post_init__(self):
        if not self.capacities or len(set(self.capacities)) != len(self.capacities) or any(q not in CAPACITIES for q in self.capacities):
            raise ValueError('Capacities must be distinct members of 4,5,6,7,8')
        if self.iterations < 1 or self.population < 2 or min(self.generations, self.evaluation_budget) < 0:
            raise ValueError('Invalid search budget')
        if self.subproblem_seconds is not None and (not math.isfinite(self.subproblem_seconds) or self.subproblem_seconds <= 0):
            raise ValueError('Invalid integer subproblem time limit')

    def costs(self, capacity):
        ratio = Decimal(capacity) / Decimal(6)
        return float(Decimal('50') * ratio), float(Decimal('0.3') * ratio)

    def stage1(self, capacity):
        purchase, operating = self.costs(capacity)
        return Stage1Params(capacity=capacity, purchase_cost=purchase, operating_cost=operating,
                            population=self.population, generations=self.generations,
                            evaluation_budget=self.evaluation_budget or None, seed=self.seed)

    def stage2(self, capacity):
        purchase, operating = self.costs(capacity)
        return Stage2Params(capacity=capacity, purchase_cost=purchase, operating_cost=operating,
                            iterations=self.iterations, seed=self.seed,
                            subproblem_seconds=self.subproblem_seconds)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def read_json(path, default=None):
    return json.loads(Path(path).read_text(encoding='utf-8')) if Path(path).exists() else default


def canonical(value):
    return json.loads(json.dumps(value))


def fingerprint(data):
    digest = hashlib.sha256()
    for file in sorted(Path(data).glob('*.csv')):
        digest.update(file.name.encode())
        digest.update(file.read_bytes())
    return digest.hexdigest()


class ProcessGroup:
    def __init__(self):
        self.lock = Lock()
        self.processes = set()
        self.cancelled = False

    def start(self, *args, **kwargs):
        with self.lock:
            if self.cancelled:
                raise RuntimeError('Capacity comparison cancelled')
            process = subprocess.Popen(*args, **kwargs)
            self.processes.add(process)
            return process

    def finished(self, process):
        with self.lock:
            self.processes.discard(process)

    def cancel(self):
        with self.lock:
            self.cancelled = True
            processes = list(self.processes)
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def map_cases(function, values, workers, abort):
    values = list(values)
    pool = ThreadPoolExecutor(max_workers=workers)
    futures = []
    try:
        futures = [pool.submit(function, value) for value in values]
        indices = {future: i for i, future in enumerate(futures)}
        results = [None] * len(futures)
        for future in as_completed(futures):
            results[indices[future]] = future.result()
        return results
    except BaseException:
        for future in futures:
            future.cancel()
        abort()
        raise
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


def execute(command, logfile, process_group=None):
    with Path(logfile).open('a', encoding='utf-8') as log:
        start = process_group.start if process_group is not None else subprocess.Popen
        process = start(command, cwd=PROJECT, stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
        try:
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(f'[{Path(logfile).parent.name}/{Path(logfile).stem}] {line}', end='', flush=True)
            return process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
        finally:
            if process.stdout is not None:
                process.stdout.close()
            if process_group is not None:
                process_group.finished(process)


def stage1_command(settings, capacity, data, folder):
    purchase, operating = settings.costs(capacity)
    return [sys.executable, '-u', '-m', 'stage1.main', '--data', str(data),
            '--output', str(folder / 'stage1.json'), '--capacity', str(capacity),
            '--purchase-cost', str(purchase), '--operating-cost', str(operating),
            '--population', str(settings.population), '--generations', str(settings.generations),
            '--evaluation-budget', str(settings.evaluation_budget), '--seed', str(settings.seed)]


def stage2_command(settings, capacity, data, folder, bounds_file, resume):
    purchase, operating = settings.costs(capacity)
    command = [sys.executable, '-u', '-m', 'stage2.main', '--data', str(data),
               '--timetables', str(folder / 'stage1.json'), '--output', str(folder / 'stage2'),
               '--capacity', str(capacity), '--purchase-cost', str(purchase),
               '--operating-cost', str(operating), '--iterations', str(settings.iterations),
               '--seed', str(settings.seed), '--objective-bounds', str(bounds_file), '--no-reports']
    if settings.subproblem_seconds is not None:
        command += ['--subproblem-seconds', str(settings.subproblem_seconds)]
    if resume:
        command += ['--resume', '--retry-failed']
    return command


def prepare_case(settings, capacity, data, folder, net, executor):
    folder.mkdir(exist_ok=True)
    state = read_json(folder / 'case.json', {})
    if not state.get('stage1_complete'):
        state.update(capacity=capacity, phase='stage1', status='running')
        write_json(folder / 'case.json', state)
        print(f'Capacity {capacity}: optimizing its own timetable set', flush=True)
        code = executor(stage1_command(settings, capacity, data, folder), folder / 'stage1.log')
        if code:
            raise RuntimeError(f'Stage 1 exited with status {code}')
    first_run = read_json(folder / 'stage1_run.json')
    if not first_run or first_run.get('parameters') != canonical(asdict(settings.stage1(capacity))):
        raise ValueError('First-stage parameters do not match the comparison design')
    tables = read_json(folder / 'stage1.json')
    if not isinstance(tables, list) or not 0 < len(tables) <= settings.stage1(capacity).candidates:
        raise ValueError('No valid representative timetables')
    for item in tables:
        errors = violations(item['timetable'], net, settings.stage1(capacity))
        if errors:
            raise ValueError(f'Illegal first-stage timetable: {errors}')
    state.update(capacity=capacity, stage1_complete=True, planned_timetables=len(tables), phase='stage1_completed', stage1_seconds=first_run['elapsed_seconds'])
    write_json(folder / 'case.json', state)
    print(f"Capacity {capacity}: stage 1 completed in {first_run['elapsed_seconds']:.2f} seconds; {len(tables)} timetables", flush=True)
    return objective_bounds(net, tables, settings.stage2(capacity))[0]


def nondominated_rows(rows):
    unique = {}
    for row in rows:
        point = (row['F1'], row['F2'])
        if not all(math.isfinite(x) and x >= 0 for x in point):
            raise ValueError('Invalid physical objectives')
        unique.setdefault(objective_key(point), row)
    items = list(unique.values())
    fronts = layers2d([(r['F1'], r['F2']) for r in items])
    return sorted((items[i] for i in fronts[0]), key=lambda r: (r['F1'], r['F2'])) if fronts else []


def collect_case(settings, capacity, folder, net, shared_bounds):

    planned = len(read_json(folder / 'stage1.json', []))
    run = read_json(folder / 'stage2/run.json', {})
    timing = read_json(folder / 'stage2/timing.json', {})
    progress = read_json(folder / 'stage2/progress.json', {})
    state = read_json(folder / 'case.json', {})
    first = read_json(folder / 'stage1_run.json', {})
    case = {'capacity': capacity, 'status': 'incomplete', 'budget_complete': False,
            'planned_timetables': planned, 'attempted_timetables': progress.get('completed', 0),
            'feasible_timetables': len(run.get('search', [])), 'front': [],
            'failed_initializations': len(progress.get('failures', [])),
            'stage1_seconds': first.get('elapsed_seconds'),
            'stage1_evaluation_calls': first.get('evaluation_calls'),
            'stage1_generations_completed': first.get('generations_completed'),
            'stage1_termination_reason': first.get('termination_reason'),
            'stage2_seconds': timing.get('total_elapsed_seconds'), 'error': state.get('error')}
    if not run:
        if planned and progress.get('completed') == planned and len(progress.get('failures', [])) == planned:
            case['status'] = 'no_feasible_solution_found'
        return case
    expected = canonical(asdict(settings.stage2(capacity)))
    configuration = read_json(folder / 'stage2/configuration.json', {})
    if run.get('parameters') != expected or configuration.get('parameters') != expected:
        raise ValueError(f'Capacity {capacity}: operational/search parameter mismatch')
    expected_scales = read_json(folder.parent / 'objective_bounds.json', {}).get('allocation_scales')
    if configuration.get('allocation_scales') != expected_scales:
        raise ValueError('Capacity allocation scales do not match')
    if shared_bounds is None or configuration.get('objective_bounds') != list(shared_bounds):
        raise ValueError('Capacity cases did not use the same search normalization')
    rows = []
    for entry in read_json(folder / 'stage2/pareto.json', []):
        name = entry['solution']
        if not name.startswith('solution_') or not name.removeprefix('solution_').isdigit():
            raise ValueError('Invalid solution filename')
        file = folder / 'stage2' / (name + '.json')
        errors = audit_file(file, net, settings.stage2(capacity))
        if errors:
            raise ValueError(f'Invalid saved solution: {errors[:3]}')
        raw = read_json(file)
        if len(raw['services']) != len(net.passengers):
            raise ValueError('Incomplete passenger service')
        f1, f2 = raw['objectives']
        if any(abs(a - b) > 1e-6 for a, b in zip((f1, f2), (entry['F1'], entry['F2']))):
            raise ValueError('Front summary/objective mismatch')
        comp = raw['components']
        wait = comp['normal_wait_minutes'] + comp['drt_wait_minutes']
        if abs(f2 - wait - comp['normal_detour_minutes']) > 1e-6:
            raise ValueError('Passenger objective decomposition mismatch')
        rows.append({'F1': f1, 'F2': f2, 'fleet': raw['metadata']['fleet_size'],
                     'waiting_minutes': wait, 'detour_minutes': comp['normal_detour_minutes'],
                     'purchase_cost': comp['purchase'], 'travel_cost': f1 - comp['purchase']})
    case['front'] = nondominated_rows(rows)
    search = run.get('search', [])
    case['complete_feasible_iterations'] = sum(s['complete_feasible_iterations'] for s in search)
    case['iterations_per_timetable'] = [s['iterations'] for s in search]
    done = (planned > 0 and progress.get('completed') == planned and not progress.get('failures')
            and len(search) == planned and timing.get('status') == 'completed'
            and state.get('stage2_exit_code') == 0 and bool(case['front']))
    case['budget_complete'] = done and all(s['iterations'] == settings.iterations for s in search)
    case['status'] = 'completed' if case['budget_complete'] else 'incomplete'
    return case


def weakly_covers(front_a, front_b):
    return bool(front_a and front_b) and all(any(
        all(x <= y for x, y in zip(objective_key((a['F1'], a['F2'])), objective_key((b['F1'], b['F2']))))
        for a in front_a) for b in front_b)


def analyze(cases):
    points = [row for case in cases for row in case['front']]
    complete = ({c['capacity'] for c in cases} == set(CAPACITIES)
                and all(c['budget_complete'] for c in cases))
    result = {'all_five_capacities_budget_complete': complete, 'recommended_capacity': None,
              'recommendation_basis': None, 'preference_free_observed_best': False,
              'cases': cases, 'F2_definition': 'normal waiting + DRT waiting + normal extra ride time; person-minutes',
              'scope': 'Empirical fronts from this search budget, not globally optimal fronts.'}
    if not points:
        result['reason'] = '尚无已审计可行点；初始化失败不能证明该容量不可行。'
        return result
    low = [min(r[f'F{j}'] for r in points) for j in (1, 2)]
    high = [max(r[f'F{j}'] for r in points) for j in (1, 2)]
    span = [b - a if b > a else 1.0 for a, b in zip(low, high)]
    result['comparison_normalization'] = {'min': low, 'max': high, 'normalized_reference': [1.1, 1.1],
                                         'method': 'One pooled observed min/max scale for all capacities; zero spans use 1.'}
    for case in cases:
        normalized = [tuple((r[f'F{j+1}'] - low[j]) / span[j] for j in (0, 1)) for r in case['front']]
        case['hypervolume'] = hypervolume(normalized, (1, 1), .1) if normalized else 0.0
    best_hv = max(c['hypervolume'] for c in cases)
    hv_best = [c['capacity'] for c in cases if c['front'] and math.isclose(c['hypervolume'], best_hv, rel_tol=1e-12, abs_tol=1e-12)]
    covering = [c['capacity'] for c in cases if c['front'] and all(
        c is other or weakly_covers(c['front'], other['front']) for other in cases)]
    result.update(highest_hv_capacities=hv_best, observed_front_covering_capacities=covering)
    if not complete:
        result['reason'] = '存在未运行容量、失败分支或未完成的迭代预算，暂不作最终推荐。'
    elif len(covering) == 1:
        result.update(recommended_capacity=covering[0], recommendation_basis='observed_front_covers_every_other_front',
                      preference_free_observed_best=True)
    elif len(hv_best) == 1:
        result.update(recommended_capacity=hv_best[0], recommendation_basis='maximum_common_normalized_hypervolume')
        result['reason'] = '这是HV准则下的折中推荐；前沿交叉不意味着存在无偏好的唯一最优容量。'
    else:
        result['reason'] = '前沿或HV存在并列，需要明确决策偏好才能选定唯一容量。'
    return result


def draw_fronts(cases, directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    fonts = {f.name for f in font_manager.fontManager.ttflist}
    font = next((f for f in ('SimSun', 'Noto Serif CJK SC', 'Microsoft YaHei', 'SimHei') if f in fonts), None)
    style = {'font.family': font or 'DejaVu Serif', 'axes.unicode_minus': False,
             'font.size': 12, 'xtick.direction': 'in', 'ytick.direction': 'in'}
    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        palette = plt.get_cmap('Blues')
        drawn = False
        for case in sorted(cases, key=lambda c: c['capacity']):
            rows = case['front']
            if not rows:
                continue
            q = case['capacity']
            ax.plot([r['F1'] for r in rows], [r['F2'] for r in rows], '-o',
                    color=palette(.90 - .13 * (q - 4)), linewidth=.9, markersize=4, label=str(q))
            drawn = True
        ax.set_xlabel('运营成本（元）' if font else 'Operating cost (yuan)')
        ax.set_ylabel('乘客时间损失（人·分钟）' if font else 'Passenger time loss (person-min)')
        if {c['capacity'] for c in cases} != set(CAPACITIES) or not all(c['budget_complete'] for c in cases):
            ax.set_title('部分结果：尚未完成全部预算' if font else 'Partial results: unfinished comparison')
        ax.ticklabel_format(axis='both', style='plain', useOffset=False)
        if drawn:
            ax.legend(title='MU 载客容量（人）' if font else 'MU capacity (persons)',
                      loc='upper right', frameon=True, fancybox=False, edgecolor='black')
        else:
            ax.text(.5, .5, 'No audited feasible points', transform=ax.transAxes, ha='center')
        ax.minorticks_on()
        fig.tight_layout()
        outputs = []
        for suffix in ('png', 'pdf'):
            path = Path(directory) / ('capacity_pareto_front.' + suffix)
            fig.savefig(path, dpi=300, bbox_inches='tight')
            outputs.append(str(path))
        plt.close(fig)
    return outputs


def describe_trend(capacities, values, label, unit):

    best = min(values)
    winners = [str(q) for q, value in zip(capacities, values) if math.isclose(value, best, rel_tol=1e-9, abs_tol=1e-7)]
    down = [b <= a + 1e-7 for a, b in zip(values, values[1:])]
    up = [b >= a - 1e-7 for a, b in zip(values, values[1:])]
    pivot = values.index(best)
    if all(down) and all(up):
        trend = '基本不变'
    elif all(down):
        trend = '随容量增大单调不增'
    elif all(up):
        trend = '随容量增大单调不减'
    elif 0 < pivot < len(values) - 1 and all(down[:pivot]) and all(up[pivot:]):
        trend = '先下降后上升'
    else:
        trend = '非单调变化，不能概括为先下降后上升'
    return f"已获得前沿的{label}{trend}；最低值{best:.2f}{unit}，对应容量{'、'.join(winners)}人。"


def write_report(root, cases, settings, search_bounds, workers=1):
    result = analyze(cases)
    result.update(settings=canonical(asdict(settings)), search_objective_bounds=search_bounds, parallel_workers=workers,
                  cost_scaling='purchase=50*capacity/6; per_km=0.3*capacity/6; maximum formation fixed at 4')
    with (root / 'pareto_fronts.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.writer(stream)
        writer.writerow(['capacity', 'F1', 'F2'])
        for case in cases:
            writer.writerows((case['capacity'], p['F1'], p['F2']) for p in case['front'])
    lines = ['本实验以6人模块为基准，购置费及公里费按容量同比例变化；最大编组始终为4。',
             f'各容量分别优化时刻表，第二阶段每套计划迭代{settings.iterations}次，初始化和修复内部保持原版。',
             '纵轴F2为等待时间与普通乘客额外乘车时间之和，不等同于纯候车时间。', '']
    for case in cases:
        front = case['front']
        if not front:
            lines.append(f"容量{case['capacity']}：尚无已审计的可行前沿，状态为{case['status']}；这不证明该容量不可行。")
            continue
        cost_point = min(front, key=lambda p: p['F1'])
        time_point = min(front, key=lambda p: p['F2'])
        q = case['capacity']
        lines.append(f"容量{q}：前沿{len(front)}个目标点，最低运营成本{cost_point['F1']:.2f}元，最低F2为{time_point['F2']:.2f}人·分钟。两端点可能来自不同方案。")
        lines.append(f"其中最低成本方案启用{cost_point['fleet']}个实体模块，购置部分{cost_point['purchase_cost']:.2f}元、行驶部分{cost_point['travel_cost']:.2f}元；该方案等待{cost_point['waiting_minutes']:.2f}、额外乘车{cost_point['detour_minutes']:.2f}人·分钟。")
        lines.append(f"该容量完成{case['feasible_timetables']}/{case['planned_timetables']}套可行时刻表，迭代次数为{case.get('iterations_per_timetable', [])}；同尺度HV={case.get('hypervolume', 0):.8f}。")
    lines.append('')
    available = sorted((c for c in cases if c['front']), key=lambda c: c['capacity'])
    if len(available) > 1:
        capacities = [c['capacity'] for c in available]
        lines.append(describe_trend(capacities, [min(p['F1'] for p in c['front']) for c in available], '最低运营成本', '元'))
        lines.append(describe_trend(capacities, [min(p['F2'] for p in c['front']) for c in available], '最低乘客时间损失', '人·分钟'))
        lines.append('以上比较前沿两端的最优观测值，不表示同一个方案同时达到两个最小值。')
    q = result['recommended_capacity']
    if q is None:
        lines.append('目前不能给出唯一最合适容量。' + result.get('reason', ''))
    elif result['preference_free_observed_best']:
        lines.append(f'在本次相同计算预算得到的经验前沿中，容量{q}的前沿覆盖其余容量的前沿，建议优先考虑该容量；这不是全局最优证明。')
    else:
        lines.append(f'容量{q}在共同归一化参考点下的HV最大，按兼顾两目标的HV准则建议选{q}人模块。前沿存在交叉，不能认定它在所有服务水平下都最好。')
    lines.append('容量增长会同时提高单模块单价与单公里费用，也可能减少模块需求；应结合上述实际分解判断，不预设成本必然先降后升。')
    lines.append('单随机种子、启发式搜索及整数求解限时可能影响前沿；需要稳健结论时应另行开展多种子重复实验。')
    (root / 'analysis.md').write_text('\n\n'.join(lines) + '\n', encoding='utf-8')
    result['figures'] = draw_fronts(cases, root)
    write_json(root / 'analysis.json', result)
    return result


def run(settings, data, output, resume=False, report_only=False, executor=execute, workers=2):
    if type(workers) is not int or not 1 <= workers <= 5:
        raise ValueError('Workers must be between 1 and 5')
    data, root = Path(data).resolve(), Path(output).resolve()
    net = load_network(data)
    if Stage1Params().max_formation != 4 or Stage2Params().max_formation != 4:
        raise ValueError('This comparison requires maximum formation fixed at 4')
    design = canonical({'version': 1, 'settings': asdict(settings), 'data_fingerprint': fingerprint(data),
                        'stage1_defaults': asdict(Stage1Params()), 'stage2_defaults': asdict(Stage2Params())})
    root.mkdir(parents=True, exist_ok=True)
    if resume or report_only:
        if read_json(root / 'configuration.json') != design:
            raise ValueError('Data/model/search settings changed; use identical arguments or a new output folder')
    elif any(root.iterdir()):
        raise ValueError('Output is not empty; use --resume or a new output folder')
    write_json(root / 'configuration.json', design)
    prior = read_json(root / 'timing.json', {}).get('total_elapsed_seconds', 0)
    started = perf_counter()
    status = 'interrupted_or_failed'
    bounds = read_json(root / 'objective_bounds.json', {}).get('upper')
    process_group = ProcessGroup()
    if executor is execute:
        executor = partial(execute, process_group=process_group)
    try:
        if not report_only:
            def prepare(q):
                folder = root / f'capacity_{q}'
                try:
                    return q, prepare_case(settings, q, data, folder, net, executor)
                except (ValueError, RuntimeError, OSError) as error:
                    state = read_json(folder / 'case.json', {})
                    state.update(capacity=q, stage1_complete=False, error=str(error))
                    write_json(folder / 'case.json', state)
                    print(f'Capacity {q}: stage 1 unavailable: {error}', flush=True)
                    return q, None
            prepared = {q: upper for q, upper in map_cases(prepare, settings.capacities, workers, process_group.cancel) if upper is not None}
            common = [max(v[j] for v in prepared.values()) for j in (0, 1)] if prepared else None
            if bounds is not None and common != bounds:
                raise ValueError('Changed shared search bounds; do not combine existing searches with a new comparison design')
            bounds = common
            if bounds is not None:
                pool = [item for q in prepared for item in read_json(root / f'capacity_{q}' / 'stage1.json')]
                allocation_scales = reference_scales(pool, bounds)
                write_json(root / 'objective_bounds.json', {'upper': bounds, 'allocation_scales': allocation_scales})
            def search(q):
                folder = root / f'capacity_{q}'
                state = read_json(folder / 'case.json', {})
                previous = collect_case(settings, q, folder, net, bounds)
                if previous['budget_complete']:
                    return
                state.update(phase='stage2', status='running')
                write_json(folder / 'case.json', state)
                print(f"Capacity {q}: stage 2, {state['planned_timetables']} timetables x {settings.iterations} iterations", flush=True)
                code = executor(stage2_command(settings, q, data, folder, root / 'objective_bounds.json',
                                                (folder / 'stage2/configuration.json').exists()), folder / 'stage2.log')
                state.update(stage2_exit_code=code, phase='stage2_completed', status='completed' if code == 0 else 'failed')
                if code:
                    state['error'] = f'Stage 2 exited with {code}; inspect stage2.log and failure.json'
                else:
                    state.pop('error', None)
                write_json(folder / 'case.json', state)
            map_cases(search, prepared, workers, process_group.cancel)
        cases = [collect_case(settings, q, root / f'capacity_{q}', net, bounds) for q in settings.capacities]
        result = write_report(root, cases, settings, bounds, workers)
        status = 'completed' if result['all_five_capacities_budget_complete'] else 'partial'
        return result
    finally:
        write_json(root / 'timing.json', {'status': status, 'invocation_elapsed_seconds': perf_counter() - started,
                                          'total_elapsed_seconds': prior + perf_counter() - started, 'parallel_workers': workers})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=PROJECT / 'data')
    parser.add_argument('--output', type=Path, default=PROJECT / 'results/capacity_sensitivity')
    parser.add_argument('--capacities', type=int, nargs='+', choices=CAPACITIES, default=CAPACITIES)
    parser.add_argument('--iterations', type=int, default=20)
    parser.add_argument('--population', type=int, default=Stage1Params().population)
    parser.add_argument('--generations', type=int, default=Stage1Params().generations)
    parser.add_argument('--evaluation-budget', type=int, default=0)
    parser.add_argument('--subproblem-seconds', type=float)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--workers', type=int, choices=range(1, 6), default=2)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--report-only', action='store_true', )
    args = parser.parse_args()
    if args.report_only:
        saved = read_json(args.output / 'configuration.json')
        if not saved:
            parser.error('No saved sensitivity configuration')
        settings = Settings(**{**saved['settings'], 'capacities': tuple(saved['settings']['capacities'])})
    else:
        settings = Settings(capacities=tuple(args.capacities), iterations=args.iterations,
                            population=args.population, generations=args.generations,
                            evaluation_budget=args.evaluation_budget, seed=args.seed,
                            subproblem_seconds=args.subproblem_seconds)
    report = run(settings, args.data, args.output, args.resume, args.report_only, workers=args.workers)
    print('Recommended capacity:', report['recommended_capacity'], flush=True)
    print('Report:', args.output / 'analysis.md', flush=True)
    print('Figure:', args.output / 'capacity_pareto_front.png', flush=True)
    if not report['all_five_capacities_budget_complete']:
        print('INCOMPLETE: no final recommendation until all five capacities finish their budgets.', flush=True)
        raise SystemExit(2)


if __name__ == '__main__':
    main()
