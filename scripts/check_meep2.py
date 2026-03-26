import meep
print("Type of meep:", type(meep))
print("meep contents:", meep)
print("File:", meep.__file__ if hasattr(meep, '__file__') else 'N/A')

# Try accessing attributes
try:
    print("\nTrying to access meep.Simulation...")
    sim = meep.Simulation
    print("Simulation:", sim)
except Exception as e:
    print(f"Error: {e}")