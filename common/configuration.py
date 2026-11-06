def search_defaults(value):
    if not isinstance(value, dict):
        return value
    result = {key: search_defaults(item) for key, item in value.items()}
    if 'initial_checks' in result and 'initial_backtracks' in result:
        result.setdefault('repair_checks', 10000)
    return result


def warehouse_expansion(saved, current):
    if not isinstance(saved, dict) or not isinstance(current, dict) or saved.keys() != current.keys():
        return saved == current
    for key, value in current.items():
        previous = saved[key]
        if key == 'depot_capacity':
            if type(previous) is not int or type(value) is not int or not 1 <= previous <= value:
                return False
        elif isinstance(value, dict):
            if not warehouse_expansion(previous, value):
                return False
        elif previous != value:
            return False
    return True
