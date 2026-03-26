# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, 'scripts')
import auto_pdk_generator as apg
from PhotonicsAI.Photon import component_type_validator

# Check Y branch type recognition
print('=== Y branch Type Recognition ===')
comp_type = component_type_validator.classify_component_type('y_branch_splitter')
print('Type:', comp_type)

device_type = apg._resolve_device_type('y_branch_splitter')
print('device_type:', device_type)

# Check if template exists
if device_type:
    print('Template exists:', device_type in apg.TEMPLATES)
    
    if device_type in apg.TEMPLATES:
        print('\nTemplate preview:')
        print(apg.TEMPLATES[device_type][:500])
else:
    print('No device_type found!')

# Check type definition
if comp_type:
    type_def = component_type_validator.get_type_definition(comp_type)
    if type_def:
        print('\nType definition:')
        print('  Ports:', type_def.get('ports', {}).get('definition'))
        print('  Params:', list(type_def.get('params', {}).keys()))
else:
    print('\nNo type definition!')

# Check gdsfactory Y branch
import gdsfactory as gf
print('\n=== gdsfactory Y branch ===')
y_branch_funcs = [n for n in dir(gf.components) if 'y' in n.lower() and 'branch' in n.lower()]
print('Y branch functions:', y_branch_funcs)

if 'y_branch' in dir(gf.components):
    print('\ny_branch exists in gdsfactory')