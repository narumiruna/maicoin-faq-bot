import json


def load_json(f: str) -> dict:
    with open(f) as fp:
        return json.load(fp)
