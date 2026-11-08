import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path
from time import time
from common.data import load_network
from common.pareto import Archive
from .alns import solve, NoFeasibleSolution
from .audit import restore
from .constraints import validate
from .obj import evaluate
from .params import Params
from .report import export


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def checked_solutions(payload, net, p):
    saved = [restore(raw) for raw in payload['solutions']]
    if not saved:
        raise ValueError('Empty resume snapshot')
    for item in saved:
        before = item.objectives
        if validate(item, net, None, p) or any(abs(a - b) > 1e-6 for a, b in zip(evaluate(item, net, p), before)):
            raise ValueError('Invalid resume snapshot')
    return saved


def check_start(root, total, start):
    if start is None:
        return
    if not 1 <= start <= total:
        raise ValueError('Starting timetable is outside the input set')
    for number in range(1, start):
        if not (root / f'timetable_{number:02}/finished.json').exists():
            raise ValueError(f'Timetable {number} is not completed; cannot skip it')


def worker(spec):
    folder = Path(spec['folder'])
    folder.mkdir(parents=True, exist_ok=True)
    number = spec['index']
    p = Params(**spec['parameters'])
    net = load_network(spec['data'])
    archive = Archive()
    for filename in spec['prior']:
        payload = json.loads(Path(filename).read_text(encoding='utf-8'))
        for raw in payload['solutions']:
            archive.add(restore(raw))
    recovered = None
    checkpoint_file = folder / 'checkpoint.json'
    if spec['resume'] and checkpoint_file.exists():
        payload = json.loads(checkpoint_file.read_text(encoding='utf-8'))
        saved = checked_solutions(payload, net, p)
        recovered = {'solutions': saved, 'state': payload['state']}
        for item in saved:
            archive.add(item)
    started = time()
    write(folder / 'worker_status.json', {'pid': os.getpid(), 'started': started, 'status': 'running'})

    def progress(message):
        print(f'[Timetable {number + 1}] {message}', flush=True)
        with (folder / 'worker.log').open('a', encoding='utf-8') as stream:
            stream.write(message + '\n')

    def checkpoint(items, phase):
        for item in items:
            item.metadata['timetable_index'] = number
        destination = folder / 'iterations' / f"iteration_{phase['iteration']:03}" if phase['phase'] == 'iteration_candidate' else folder
        if 'state' in phase:
            write(checkpoint_file, {'solutions': [item.to_dict() for item in items], 'state': phase['state']})
        export(items, net, p, destination, {'timetable_index': number, 'checkpoint': {key: value for key, value in phase.items() if key != 'state'}}, tables=False)

    try:
        progress('Search started')
        solutions, stat = solve(net, spec['timetable'], p, progress=progress, checkpoint=checkpoint,
                                resume=recovered, global_archive=archive, upper=tuple(spec['upper']),
                                allocation_scales=tuple(spec['scales']))
    except NoFeasibleSolution as error:
        write(folder / 'failure.json', {'timetable': number, 'error': str(error), **error.diagnostics})
        status = 'initialization_failed'
        progress('Initialization failed; see failure.json')
    else:
        export(solutions, net, p, folder, {'timetable_index': number, 'phase': 'completed', 'search': stat}, tables=False)
        write(folder / 'finished.json', {'solutions': [item.to_dict() for item in solutions], 'search': stat})
        (folder / 'failure.json').unlink(missing_ok=True)
        status = 'completed'
        progress('Search completed')
    write(folder / 'worker_status.json', {'pid': os.getpid(), 'started': started, 'finished': time(), 'status': status})
    return number


def run_parallel(candidates, p, net, data, root, upper, scales, workers, resume, retry_failed, start=None, progress=None):
    check_start(root, len(candidates), start)
    prior = []
    pending = []
    for index, candidate in enumerate(candidates):
        folder = root / f'timetable_{index + 1:02}'
        cp = replace(p, seed=p.seed + index)
        finished = folder / 'finished.json'
        if finished.exists():
            checked_solutions(json.loads(finished.read_text(encoding='utf-8')), net, cp)
            prior.append(str(finished.resolve()))
            continue
        if resume and not retry_failed and (folder / 'failure.json').exists() and not (folder / 'checkpoint.json').exists():
            continue
        pending.append({'index': index, 'timetable': candidate['timetable'], 'parameters': asdict(cp),
                        'folder': str(folder.resolve()), 'data': str(Path(data).resolve()), 'upper': upper,
                        'scales': scales, 'resume': resume})
    if not pending:
        return
    for spec in pending:
        spec['prior'] = prior
    pool = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'))
    futures = []
    try:
        futures = [pool.submit(worker, spec) for spec in pending]
        for future in as_completed(futures):
            index = future.result()
            if progress:
                progress(index)
    except BaseException:
        for future in futures:
            future.cancel()
        for process in list((getattr(pool, '_processes', None) or {}).values()):
            if process.is_alive():
                process.terminate()
        pool.shutdown(wait=True, cancel_futures=True)
        raise
    else:
        pool.shutdown(wait=True)
