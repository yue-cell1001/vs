import csv
import json
from pathlib import Path
from .model import Network, Node, Passenger

def load_network(directory: str | Path) -> Network:
    directory = Path(directory)

    def read(name):
        with (directory / name).open(encoding='utf-8-sig', newline='') as f:
            return list(csv.DictReader(f))
    nodes = {int(r['id']): Node(int(r['id']), int(r['type']), r['name'], float(r['x']), float(r['y'])) for r in read('nodes_with_drt.csv')}
    lines = {r['RouteID_Str']: tuple(map(int, r['stop_sequence'].split(';'))) for r in read('lines.csv') if not r['RouteID_Str'].startswith('Virtual_')}
    edges = {(int(r['from_id']), int(r['to_id'])): (float(r['distance']) / 1000, float(r['travel_time'])) for r in read('edges.csv')}
    passengers = [Passenger(int(r['id']), int(r['origin']), int(r['dest']), int(r['type']), float(r['tw_start']), float(r['tw_end']), r['line_id'], int(r['origin_transfer_station_id']) if r.get('origin_transfer_station_id') else None, int(r['destination_transfer_station_id']) if r.get('destination_transfer_station_id') else None) for r in read('passengers.csv')]
    depots = {}
    for n in nodes.values():
        if n.kind == 1:
            matches = [d.id for d in nodes.values() if d.kind == 2 and (d.x, d.y) == (n.x, n.y)]
            if len(matches) != 1:
                raise ValueError(f'Interchange {n.id}: expected one colocated depot')
            depots[n.id] = matches[0]
    net = Network(nodes, lines, edges, passengers, depots)
    net.line_families = {line: line.rsplit('_', 1)[0] if line.endswith(('_up', '_down')) else line for line in lines}
    mapping = directory / 'drt_transfer_station_mapping.csv'
    if mapping.exists():
        for row in read(mapping.name):
            net.transfer_candidates[int(row['drt_id'])] = {'pickup': json.loads(row['pickup_transfer_station_candidates']), 'dropoff': json.loads(row['dropoff_transfer_station_candidates'])}
    for line, stops in lines.items():
        for a, b in zip(stops, stops[1:]):
            if (a, b) not in edges:
                raise ValueError(f'Missing directed road edge {line}: {a}->{b}')
    if len({p.id for p in passengers}) != len(passengers):
        raise ValueError('Duplicate passenger IDs')
    for p in passengers:
        if p.origin not in nodes or p.dest not in nodes or p.earliest > p.latest:
            raise ValueError(f'Invalid passenger {p.id}')
        if not p.kind:
            stops = lines[p.line]
            if stops.index(p.origin) >= stops.index(p.dest):
                raise ValueError(f'Invalid directional OD for passenger {p.id}')
    return net
