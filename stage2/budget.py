from time import monotonic

class SearchTimeout(RuntimeError):
    pass

def check(decoder):
    deadline = getattr(decoder, 'deadline', None)
    if deadline is not None and monotonic() >= deadline:
        raise SearchTimeout('Search time budget exhausted')
