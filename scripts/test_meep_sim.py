import sys
sys.path.insert(0, '.')
from PhotonicsAI.Photon import meep_runner

print("Running Y-branch Meep simulation...")
result = meep_runner.run_meep_simulation(
    "y_branch",
    {"gap": 2.0, "width": 0.5, "length": 30.0},
    "build"
)
print("Result:", result)