import json
from pathlib import Path
import random

INPUT_DIR = Path('/usr/src/app/InputData')
OUTPUT_DIR = Path('/usr/src/app/output')
DATA_DIR = Path('.')


labels = {}
for i, x in enumerate((INPUT_DIR / "test").iterdir()):
    labels[x.name] = random.randint(0, 1)
    
(Path(OUTPUT_DIR)/'labels').write_text(json.dumps(labels))
