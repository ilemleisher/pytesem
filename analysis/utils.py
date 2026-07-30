from operator import itemgetter


def unpack(reduced_data, keys):
    try:
        return itemgetter(*keys)(reduced_data)
    except KeyError as e:
        raise KeyError(f"Key {e} not found in reduced_data.") from e