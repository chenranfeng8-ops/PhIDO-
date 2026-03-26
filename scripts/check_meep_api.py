import meep as mp
print("Meep loaded successfully")
print("Available classes and functions:")
attrs = [x for x in dir(mp) if not x.startswith('_')]
for a in attrs[:30]:
    print(f"  - {a}")

# Check important classes
important = ['Medium', 'Block', 'Cylinder', 'Simulation', 'Source', 'Vector3', 'PML', 'FluxRegion']
print("\nImportant classes:")
for cls in important:
    available = "YES" if hasattr(mp, cls) else "NO"
    print(f"  {cls}: {available}")