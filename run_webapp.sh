#!/bin/bash
export PYTHONPATH=/mnt/c/Users/PC/Desktop/PhIDO-:$PYTHONPATH
source /mnt/c/Users/PC/Desktop/PhIDO-/.venv/bin/activate
cd /mnt/c/Users/PC/Desktop/PhIDO-
streamlit run PhotonicsAI/Photon/webapp.py --server.port 8501 --server.headless true