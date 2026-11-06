from .reference import ReferenceDecoder

class Evaluator:

    def __init__(self, net, p):
        self.decoder = ReferenceDecoder(net, p)

    def __call__(self, timetable):
        return self.decoder.decode(timetable).objectives

    def detail(self, timetable):
        return self.decoder.decode(timetable).summary()

def evaluate(timetable, net, p):
    return Evaluator(net, p)(timetable)
