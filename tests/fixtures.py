from common.model import Node, Network, Passenger

def network(passengers=()):
    coordinates = {0: (0, 0), 1: (500, 0), 2: (1000, 0), 3: (1500, 0), 4: (500, -500), 5: (500, 500), 6: (500, 1000), 10: (0, 0), 11: (1000, 100), 12: (100, 300), 13: (900, 300), 14: (200, 50), 15: (800, 50), 20: (500, 0), 21: (1000, 0)}
    nodes = {k: Node(k, 2 if k in (20, 21) else 1 if k in (1, 2) else 3 if 10 <= k <= 15 else 0, str(k), x, y) for k, (x, y) in coordinates.items()}
    lines = {'A': (0, 1, 2, 3), 'Ar': (3, 2, 1, 0), 'B': (4, 1, 5, 6), 'Br': (6, 5, 1, 4)}
    edges = {}
    for stops in lines.values():
        for a, b in zip(stops, stops[1:]):
            edges[a, b] = (0.5, 1.0)
    return Network(nodes, lines, edges, list(passengers), {1: 20, 2: 21})

def timetable():
    return {line: [0, 5, 10, 15] for line in ('A', 'Ar', 'B', 'Br')}

def scenario(ordinary=1, drt=1):
    from dataclasses import replace
    from stage2.params import Params
    people = [Passenger(i + 1, 0, 3, 0, 0, 20, 'A') for i in range(ordinary)]
    people += [Passenger(100 + i, 12 if i == 0 else 200 + 2 * i, 13 if i == 0 else 201 + 2 * i, 1, 0, 20, '') for i in range(drt)]
    net = network(people)
    net.nodes[12] = replace(net.nodes[12], x=500, y=1000)
    net.nodes[13] = replace(net.nodes[13], x=1000, y=1000)
    for i in range(1, drt):
        net.nodes[200 + 2 * i] = replace(net.nodes[12], id=200 + 2 * i)
        net.nodes[201 + 2 * i] = replace(net.nodes[13], id=201 + 2 * i)
    return (net, Params(start=0, end=15, stop_time=0, operation_time=0, iterations=2, subproblem_seconds=1), timetable())
