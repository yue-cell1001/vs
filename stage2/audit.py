import argparse
import json
from dataclasses import replace
from pathlib import Path
from common.data import load_network
from common.model import Visit, Leg, Duty, Ride, Service, Excursion, Solution
from common.timetable import make_trips
from .params import Params
from .constraints import validate
from .resources import inventory
from .obj import evaluate

def restore(raw):
    duties = {key: Duty(value['id'], [Visit(**v) for v in value['visits']], [Leg(**v) for v in value['legs']], value['vehicle']) for key, value in raw['duties'].items()}
    services = {int(key): Service(value['passenger'], tuple((Ride(**r) for r in value['rides'])), value['pickup'], value['dropoff'], value['detour'], value.get('offroute_minutes', 0)) for key, value in raw['services'].items()}
    excursions = tuple((Excursion(**{**e, 'events': tuple((tuple(x) for x in e['events']))}) for e in raw['excursions']))
    return Solution(duties, services, excursions, tuple(raw['objectives']), raw['components'], raw['errors'], raw['metadata'])

def audit_file(filename, net, p, timetable=None):
    solution = restore(json.loads(Path(filename).read_text()))
    timetable = timetable or solution.metadata['timetable']
    errors = validate(solution, net, make_trips(net, timetable, p, solution.metadata.get('operations', ())), p)
    inventory(solution.metadata['storage'], net, p, solution.metadata.get('activations', ()))
    saved = solution.objectives
    actual = evaluate(solution, net, p)
    if any((abs(a - b) > 1e-06 for a, b in zip(saved, actual))):
        errors.append('Objective recomputation mismatch')
    return errors

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('solution')
    parser.add_argument('--data', default='data')
    parser.add_argument('--run', )
    args = parser.parse_args()
    run = Path(args.run) if args.run else Path(args.solution).parent / 'run.json'
    p = Params(**json.loads(run.read_text())['parameters'])
    errors = audit_file(args.solution, load_network(args.data), p)
    if errors:
        raise SystemExit('\n'.join(errors))
    print('PASS: all passengers, physical constraints, inventory and both objective values')
if __name__ == '__main__':
    main()
