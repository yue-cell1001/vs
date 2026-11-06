from heapq import heappush, heappop
from collections import defaultdict

class RoadPaths:

    def __init__(self, net, p):
        self.net, self.p = (net, p)
        self.graph = defaultdict(list)
        for (a, b), (distance, travel) in net.edges.items():
            self.graph[a].append((b, distance, travel))
        self.cache = {}

    def path(self, start, end):
        key = (start, end)
        if key in self.cache:
            return self.cache[key]
        heap = [(0, 0, start, ())]
        seen = set()
        while heap:
            t, d, node, path = heappop(heap)
            if node in seen:
                continue
            seen.add(node)
            if node == end:
                self.cache[key] = (t, d, path)
                return (t, d, path)
            for b, dist, travel in self.graph[node]:
                heappush(heap, (t + travel, d + dist, b, path + ((node, b, dist, travel),)))
        self.cache[key] = None
        return None

    def reposition(self, start, end, budget):
        quickest = self.path(start, end)
        if quickest is None or quickest[0] > budget + 1e-07:
            return None
        key = ('distance', start, end)
        if key not in self.cache:
            heap = [(0.0, 0.0, start, ())]
            seen = set()
            answer = None
            while heap:
                distance, time, node, path = heappop(heap)
                if node in seen:
                    continue
                seen.add(node)
                if node == end:
                    answer = (time, distance, path)
                    break
                for nxt, d, t in self.graph[node]:
                    heappush(heap, (distance + d, time + t, nxt, path + ((node, nxt, d, t),)))
            self.cache[key] = answer
        shortest = self.cache[key]
        if shortest is not None and shortest[0] <= budget + 1e-07:
            return shortest
        heap = [(0.0, 0.0, start, ())]
        labels = defaultdict(list)
        while heap:
            distance, time, node, path = heappop(heap)
            if time > budget + 1e-07:
                continue
            if any((d <= distance and t <= time for d, t in labels[node])):
                continue
            labels[node] = [(d, t) for d, t in labels[node] if not (distance <= d and time <= t)] + [(distance, time)]
            if node == end:
                return (time, distance, path)
            for nxt, d, t in self.graph[node]:
                heappush(heap, (distance + d, time + t, nxt, path + ((node, nxt, d, t),)))
        return None
