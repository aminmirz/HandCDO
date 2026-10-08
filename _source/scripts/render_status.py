"""Read only completed, original-material renders in the Gamma workspace."""
import json
from pathlib import Path

work=Path(__file__).resolve().parents[1]
completed=[]
for path in sorted((work/'renders').glob('*/manifest.json')):
    try:info=json.loads(path.read_text())
    except (ValueError,OSError):continue
    if info.get('materials')=='Original generation_v2 materials; no palette substitutions':
        completed.append(info['study'])
try:state=json.loads((work/'final-status.json').read_text())
except (ValueError,OSError):state={}
print(json.dumps({'completed':completed,'batch':state}))
