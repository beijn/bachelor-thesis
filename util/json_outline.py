#!/bin/python


import json

def json_outline(j, indent = 0):
  """Returns a string with the outline of a JSON object (object keys, list lengths, terminal types)."""
  if isinstance(j, dict):
    if len(j) == 0: return "{}"
    return ''.join([f"\n{' '*indent}{k}:  " + json_outline(v, indent + 2)
                    for k, v in j.items()])

  elif isinstance(j, list):
    if len(j) == 0: return "[]"
    return f"\n{' '*indent}{len(j)} × ["\
         + json_outline(j[0], indent + 2)\
         + (f"\n{' '*indent}" if isinstance(j[0], list) or isinstance(j[0], dict) else '')\
         + "]"

  return f"{j} ({f'{type(j)}'[8:-2]})"

if __name__ == '__main__':
  from sys import argv
  for file in argv[1:]:
    print(f'  ===  {file}\n', json_outline(json.load(open(file))), '\n')