import os, shutil

def __dirs__(dirs):
  if isinstance(dirs, str):
    return dirs.split(' ')
  if isinstance(dirs, list):
    return dirs
  if isinstance(dirs, dict):
    return [os.path.join(k,sub) for k,v in dirs.items() for sub in __dirs__(v)]

def mk_cache(root, dirs=[], clear=False):
  root = os.path.join(os.path.expanduser('~'), '.cache', 'thesis', *root.split('/'))
  os.makedirs(root, exist_ok=True)
  dirs = __dirs__({root: dirs})
  for d in dirs:
    if clear:
      shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d, exist_ok=True)
  return root