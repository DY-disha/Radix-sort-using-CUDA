#!/usr/bin/env python3
"""Run from futhark-baseline after compiling sort_u32."""
import pathlib
import re
import subprocess


def parse(text):
    text = text.strip()
    if not text.startswith('[') or not text.endswith(']'):
        raise ValueError('Expected a one-dimensional u32 array')

    body = text[1:-1].strip()
    if not body:
        return []

    result = []
    for token in body.split(','):
        if not re.fullmatch(r'\s*\d+u32\s*', token):
            raise ValueError('Unexpected output token: ' + token[:80])
        result.append(int(token.strip()[:-3]))

    return result


passed = 0

for n in (10000, 100000, 1000000):
    for pattern in (
        'random', 'all_equal', 'few_distinct', 'sorted', 'reversed'
    ):
        name = f'{n}_{pattern}'

        with open(f'data/{name}.in') as inp:
            run = subprocess.run(
                ['./sort_u32'],
                stdin=inp,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )

        if run.returncode:
            raise RuntimeError(f'{name}: {run.stderr}')

        actual = parse(run.stdout)
        expected = parse(
            pathlib.Path(f'data/{name}.expected').read_text()
        )

        if len(actual) != n or actual != expected:
            raise RuntimeError(
                f'{name}: output differs from CPU std::sort'
            )

        print(f'PASS: {name}', flush=True)
        passed += 1

print(f'VALID: all {passed} datasets match CPU std::sort.')
