"""scrub_ids.py: take personal and machine identifiers out of the site's files (16 Nov 2026, Nic: "remove identifiers").

What it changes, in every .js / .json / .html / .mjs / .md file under the folders given (default: tactical/ and archived_sites/):
  1. BOXRR-23 player and recording IDs (36-character UUIDs) become neutral labels: player-0001, rec-0001, ... The SAME UUID gets the
     SAME label in every file, so the pages' own matches (e.g. the stage-9 window found in traces/index.js by user + recording +
     segment) still work. The 8-character short forms in notes ("2bbe12b6...") get the matching label too.
  2. Machine paths (the team PC's /srv data paths, ~/Downloads/..., nic_scratch/...) become plain descriptions.
It leaves model and dataset checksums (sha256 ...) alone: they identify our files, not people.
Re-run it after regenerating any data file. Run from k8_3d/:   python3 tactical/tools/scrub_ids.py
"""
import re, sys, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]                     # k8_3d/
DIRS = [ROOT / d for d in (sys.argv[1:] or ['tactical', 'archived_sites'])]
EXT = {'.js', '.json', '.html', '.mjs', '.md'}
SKIP = {'scrub_ids.py'}
UUID = re.compile(r'\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b')

files = sorted(p for d in DIRS if d.exists() for p in d.rglob('*') if p.is_file() and p.suffix in EXT and p.name not in SKIP)

# pass 1: every UUID, with its role from the key in front of it (user/player -> player, recording -> rec, a list of users -> player)
role, order = {}, []
for p in files:
    s = p.read_text(errors='ignore')
    for m in UUID.finditer(s):
        u = m.group(0); before = s[max(0, m.start() - 40):m.start()]
        if re.search(r'recording"?\s*[:=]\s*"?$', before): r = 'rec'
        elif re.search(r'(user|uid|player)"?\s*[:=]\s*"?$', before): r = 'player'
        elif p.name == 'pull_counts.json': r = 'player'                   # its usable_users list
        else: r = None
        if u not in role: role[u] = r; order.append(u)
        elif role[u] is None and r: role[u] = r
n = {'player': 0, 'rec': 0, 'id': 0}; label = {}
for u in order:
    r = role[u] or 'id'; n[r] += 1; label[u] = f'{r}-{n[r]:04d}'
short = {u[:8]: lab for u, lab in label.items()}                         # the 8-character short forms

PATHS = [
    (re.compile(r'pc:/srv/\S*results\.sqlite \(mode=ro\), tars of '), "the team's data PC: the pipeline's results database and, from the "),
    (re.compile(r'pc:/srv/[^"\'\s)]*'), "the team's data PC"),
    (re.compile(r'/srv/[a-z_]+(/[^"\'\s)]*)?'), "the team's data PC"),
    (re.compile(r'~/Downloads/abc_moredata_final/'), ''),
    (re.compile(r'`?~/Downloads/[^`"\'\s)]*`?'), 'a local downloads folder'),
    (re.compile(r'`?~/Downloads`?'), 'a local downloads folder'),
    (re.compile(r'nic_scratch/viz/k8_3d/'), ''),
    (re.compile(r'nic_scratch/viz/k8_3d'), 'k8_3d'),
    (re.compile(r'nic_scratch/data/users'), 'a local folder'),
    (re.compile(r'/Users/[a-z]+/[^"\'\s)]*'), 'a local folder'),
]
changed = 0
for p in files:
    s0 = s = p.read_text(errors='ignore')
    s = UUID.sub(lambda m: label[m.group(0)], s)
    s = re.sub(r'\b([0-9a-f]{8})(…|\.\.\.)', lambda m: short[m.group(1)] if m.group(1) in short else m.group(0), s)
    for rx, rep in PATHS: s = rx.sub(rep, s)
    if s != s0: p.write_text(s); changed += 1; print('scrubbed', p.relative_to(ROOT))
print(f'{len(label)} identifiers -> labels ({n["player"]} players, {n["rec"]} recordings, {n["id"]} other); {changed} files changed')
left = [str(p.relative_to(ROOT)) for p in files if UUID.search(p.read_text(errors='ignore'))]
if left: sys.exit('UUIDs still present in: ' + ', '.join(left))
