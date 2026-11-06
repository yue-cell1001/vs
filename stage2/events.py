class EventNetwork:

    def __init__(self):
        self.nodes = {}
        self.edges = []

    def node(self, key):
        if key not in self.nodes:
            self.nodes[key] = len(self.nodes)
        return self.nodes[key]

    def difference(self, a, b, lower=None, upper=None):
        i, j = (self.node(a), self.node(b))
        if upper is not None:
            self.edges.append((i, j, float(upper)))
        if lower is not None:
            self.edges.append((j, i, -float(lower)))

    def equal(self, a, b, value=0):
        self.difference(a, b, value, value)

    def bound(self, key, lower, upper):
        self.difference(('zero',), key, lower, upper)

    def solve(self):
        n = len(self.nodes)
        dist = [0.0] * n
        for iteration in range(n):
            changed = False
            for i, j, c in self.edges:
                if dist[j] > dist[i] + c + 1e-07:
                    dist[j] = dist[i] + c
                    changed = True
            if not changed:
                zero = dist[self.node(('zero',))] if ('zero',) in self.nodes else 0
                return {key: dist[i] - zero for key, i in self.nodes.items()}
        raise ValueError('Inconsistent event-time dependencies')

def verify_context(net, p, trips, duties):
    graph = EventNetwork()
    for key, duty in duties.items():
        for i, visit in enumerate(duty.visits):
            a = (key, i, 'arrival')
            d = (key, i, 'departure')
            graph.bound(a, visit.arrival, visit.arrival)
            graph.bound(d, visit.departure, visit.departure)
            graph.difference(a, d, lower=0)
            if visit.trip in trips:
                target = next((v for v in trips[visit.trip].visits if v.node == visit.node), None)
                if target is not None:
                    graph.bound(d, target.departure, target.departure)
            if visit.kind in ('pickup', 'dropoff'):
                graph.difference(a, d, lower=p.stop_time)
            if visit.kind in ('join', 'stored_join'):
                graph.difference(a, d, lower=p.operation_time)
        for i, leg in enumerate(duty.legs):
            duration = net.road(leg.start, leg.end)[1] if leg.kind in ('fixed', 'solo', 'empty') and (leg.start, leg.end) in net.edges else net.distance(leg.start, leg.end) / p.speed * 60
            graph.difference((key, i, 'departure'), (key, i + 1, 'arrival'), lower=duration, upper=duration)
    return graph.solve()
