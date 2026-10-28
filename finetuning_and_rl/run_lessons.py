"""Run offline, single-process lessons; never download external models."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
EXCLUDED = {'04_huggingface_sft.py', '05_reload_and_generate.py', '07_distributed_training.py'}
STDLIB = {'01_python_and_math.py', '01_bandits.py', '02_mdp_value_iteration.py',
          '03_q_learning_sarsa.py', '07_monte_carlo.py', '05_evaluation_and_uncertainty.py'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--stdlib-only', action='store_true')
    args = parser.parse_args()
    paths = [p for p in sorted(ROOT.glob('0[1-5]*/*.py')) if p.name not in EXCLUDED]
    if args.stdlib_only:
        paths = [p for p in paths if p.name in STDLIB]
    failures = []
    for path in paths:
        print(f'\n>>> {path.relative_to(ROOT)}', flush=True)
        if args.list:
            continue
        try:
            result = subprocess.run([sys.executable, str(path)], cwd=ROOT, timeout=180)
            if result.returncode:
                failures.append(str(path.relative_to(ROOT)))
        except subprocess.TimeoutExpired:
            failures.append(str(path.relative_to(ROOT))+' (timeout)')
    if not args.list:
        print(f'\n{len(paths)-len(failures)}/{len(paths)} lessons passed.')
        if failures:
            print('Failures:', *failures, sep='\n')
            raise SystemExit(1)

if __name__ == '__main__':
    main()
