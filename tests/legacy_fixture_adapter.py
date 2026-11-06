from stage2.params import Params as ModelParams

def Params(**kwargs):
    return ModelParams(**{"start": 0, "end": 15, **kwargs})
