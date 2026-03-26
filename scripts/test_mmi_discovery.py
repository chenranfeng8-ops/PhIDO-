# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, 'scripts')
import auto_pdk_generator as apg

print('Testing: 1x2 MMI')
print('='*60)

result = apg.discover_and_generate('1x2 MMI', max_papers=3)

print()
print('Result:')
print('  Device type:', result.get('device_type'))
print('  Papers found:', result.get('papers_found'))
print('  Parameters:', result.get('params'))
print('  File:', result.get('filepath'))
print('  Error:', result.get('error'))