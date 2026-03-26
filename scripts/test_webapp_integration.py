# Test webapp integration
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PhotonicsAI.Photon import component_type_validator

# Test classify
result = component_type_validator.validate_component(
    'ring_modulator_high_speed',
    ports='2x2',
    params={'radius': 15.0, 'gap': 0.25, 'length': 200.0}
)

print('Validation Result:')
print('  Component: ring_modulator_high_speed')
print('  Type:', result.get('component_type'))
print('  Valid:', result.get('valid'))
if result.get('param_check'):
    pc = result['param_check']
    print('  Params:', pc.get('params'))
    print('  Warnings:', pc.get('warnings', [])[:2])

# Test with auto_pdk_generator integration
print('\n--- auto_pdk_generator Integration ---')
import sys
sys.path.insert(0, 'scripts')
import auto_pdk_generator as apg

# Test that type_validator is accessible
print('  type_validator accessible:', hasattr(apg, 'type_validator'))
print('  classify function:', hasattr(apg.type_validator, 'classify_component_type'))

# Test classification
test_type = apg.type_validator.classify_component_type('mzi_2x2_heater')
print('  mzi_2x2_heater ->', test_type)