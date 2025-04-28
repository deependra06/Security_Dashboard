import os

# Base directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB_DIR = os.path.join(BASE_DIR, 'web')

# Storage directories
UPLOADS_DIR = os.path.join(WEB_DIR, 'uploads')
SNAPSHOTS_MOTION_DIR = os.path.join(WEB_DIR, 'snapshots', 'motion')
SNAPSHOTS_UNKNOWN_DIR = os.path.join(WEB_DIR, 'snapshots', 'unknown_faces')

# Create directories if they don't exist
for directory in [UPLOADS_DIR, SNAPSHOTS_MOTION_DIR, SNAPSHOTS_UNKNOWN_DIR]:
    os.makedirs(directory, exist_ok=True) 