import statistics
import sys

build, edit, suite, cut2, optout = {}, {}, {}, {}, None
for line in open(sys.argv[1]).read().splitlines():
    f = line.split()
    if not f:
        continue
    kv = dict(x.split('=', 1) for x in f if '=' in x)
    if f[0] == 'BUILD':
        build[f[1]] = kv
    elif f[0] == 'EDIT':
        edit[f[1]] = kv
    elif f[0] == 'SUITE':
        suite.setdefault(f[1], []).append(kv)
    elif f[0] == 'CUT2':
        cut2.setdefault(f[1], []).append(kv)
    elif f[0] == 'OPTOUT':
        optout = kv


def med(v, key):
    xs = [float(k[key]) for k in suite.get(v, []) if k.get(key, 'none') != 'none']
    return statistics.median(xs) if xs else float('nan')


print('V  build    ce      te      run1   run2   run3   run4   run5   median  r(V)   cpu-med  cpu-ratio  exits      retried')
for v in 'ACG':
    runs = suite.get(v, [])
    t = [k['took'] for k in runs] + ['-'] * (5 - len(runs))
    m, c = med(v, 'took'), med(v, 'cpu')
    r = m / med('A', 'took')
    cr = c / med('A', 'cpu')
    b = build.get(v, {}).get('build', '-')
    e = edit.get(v, {})
    print(f"{v}  {b:>7} {e.get('ce', '-'):>7} {e.get('te', '-'):>7} " + ' '.join(f'{x:>6}' for x in t[:5])
          + f"  {m:6.2f}  {r:5.3f}  {c:7.2f}  {cr:6.3f}    {','.join(k['exit'] for k in runs):10} {','.join(k['retried'] for k in runs)}")
if optout:
    a = float(build['A']['build'])
    print(f"optout {optout['optout']} s, {float(optout['optout']) / a:.3f} of build(A), compiled {optout.get('compiled')}")
for v in 'ACG':
    if v in build:
        print(f"compiled in build({v}): {build[v].get('compiled')}")
for f, runs in cut2.items():
    print('cut2', f, 'cpu median', statistics.median(float(k['cpu']) for k in runs),
          'wall median', statistics.median(float(k['wall']) for k in runs), 'exits', ','.join(k['exit'] for k in runs))
