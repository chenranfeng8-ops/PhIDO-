# Check Tidy3D web functions
from tidy3d import web
import inspect

print('web.run signature:')
sig = inspect.signature(web.run)
for name, param in sig.parameters.items():
    default = param.default if param.default != inspect.Parameter.empty else "required"
    print(f'  {name}: {default}')

print('\nAll web functions:')
funcs = [x for x in dir(web) if not x.startswith('_') and callable(getattr(web, x))]
for f in funcs:
    print(f'  - {f}')

# Check for local run
if hasattr(web, 'run_local'):
    print('\nLOCAL RUN AVAILABLE!')
elif hasattr(web, 'api'):
    api = web.api
    print(f'\nweb.api attributes: {[x for x in dir(api) if not x.startswith("_")]}')