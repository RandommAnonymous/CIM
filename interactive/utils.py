import json
from pathlib import Path
import re

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle)


def get_conjunctions(singular_captions, human_caption):
    singular = human_caption in singular_captions
    verb = "is" if singular else "are"
    return verb

def simplify_caption(caption):
    caption = caption.strip('.').lower()
    return caption


def int_to_en(integer):
    d = {0: 'zero', 1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five',
         6: 'six', 7: 'seven', 8: 'eight', 9: 'nine', 10: 'ten'}
    return d[integer]