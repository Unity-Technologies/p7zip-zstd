#!/usr/bin/env python3
"""
7za extraction benchmarking script for Linux

Copyright (c)2019 Unity Technologies A/S

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import os
import sys

# Build settings (will autobuild variants with different options for benchmarking)
makeroot = 'p7zip_16.02'
makefile = 'p7zip_16.02/makefile.machine'
maketarget = '7za'
maketarget_path = 'p7zip_16.02/bin/7za'
buildvariants_path = 'bench-bin'

test_options = [
    #'-Os', # terribly slow, kinda pointless
    #'-O1', # rather slow, kinda pointless
    #'-O2',
    #'-O3',
    #'-O2 -pie -fpie',
    #'-O3 -pie -fpie',
    #'-O2 -flto',
    #'-O3 -flto',
    #'-O2 -flto -pie -fpie',
    '-O3 -flto -pie -fpie',
    #'-O2 -flto -D_FORTIFY_SOURCE=0 -fno-stack-protector -no-pie -fno-pie', # evaluate cost of improved security
    #'-O3 -flto -D_FORTIFY_SOURCE=0 -fno-stack-protector -no-pie -fno-pie', # evaluate cost of improved security
]
test_compilers = [
    #('clang', 'clang++', test_options),
    #('gcc-8', 'g++-8', test_options),
    #('gcc-7', 'g++-7', ['-O2 -flto -pie -fpie']),
]

# Prebuilt/external variants that should (also) be benchmarked ( {'friendly name': 'path'} )
test_prebuilt = {
    'u7za': '/data/u/unity/External/7z/osx/7za',
    'yamato': '/tmp/p7zip/7za',
}


# Benchmarking settings

# TODO: place paths to one or more representative archive files here
# Each build or prebuilt variant will be tested with each of these files.
prefix = '~/Library/Caches/Stevedore/artifacts/' if sys.platform == 'darwin' else '~/.cache/Stevedore/artifacts/'
test_files = [
    os.path.expanduser(prefix + name)
    for name in (
        'tundra-linux-x64/fd4be3915562_af08ccc59497adbb24ca5cfb656b331fc1cf5dd5ff5cb84bcc45e1c1e801cfd8.zip',
        'referenceassemblies_v46/1_3750434ae503231abc3db765a9a8972337bdb0392b992cb22f879fe42f62c746.7z',
        'android-ndk-linux-x86_64/r16b_bcdea4f5353773b2ffa85b5a9a2ae35544ce88ec5b507301d8cf6a76b765d901.7z',
    )
]

raw_results_file = 'benchresults.json'
sample_count = 20
sample_repetitions = 100 # executions per sample

# If using test files of varying size (and processing time), uncomment
# the following to repeat for a given time instead of a fixed number.
sample_repetitions = None
sample_time = 20 # seconds

# Desired confidence in (statistical significance of) final results.
confidence_two_sided = 0.95


# -----------------------------------------------------------------------------

import collections
import contextlib
import datetime
import json
import math
import os
import random
import re
import shlex
import subprocess
import statistics
import time
import urllib.parse
import zlib


_size_units = [
    (1000000000, ' GB'),
    (1000000, ' MB'),
    (1000, ' kB'),
]

def friendly_size(size):
    for unit in _size_units:
        if size >= unit[0]: break
    scale, suffix = unit
    return str((size + scale - 1) // scale) + suffix



class BenchResultBase:
    def __str__(self):
        return ' '.join(f'{field}={getattr(self, field):{fmt}}' for field, fmt in zip(self._fields, self._formats))

    @classmethod
    def reduce(cls, func, sample_list):
        return cls(*(
            func(sample[i] for sample in sample_list)
            for i in range(len(cls._fields))
        ))



if sys.platform == 'linux':
    class BenchResult(collections.namedtuple('BenchResult', '''
        t_sys t_user t_wall
        n_ctxswitch n_swapout n_hardfault n_softfault
        avg_unshared avg_total avg_rss max_rss
    '''), BenchResultBase):
        _formats    = '6.2f 6.2f 6.2f  5.2f .1f .1f 6.1f  .0f .0f .0f .0f'.split()
        _time_format = '%S   %U   %e    %c   %W  %F  %R    %D  %K  %t  %M'
        time_format_durations = 3 # the first 3 numbers are durations (in seconds), to be scaled by 1000 (to ms)
        time_format_per_rep = 7 # the first 7 numbers should be divided by number of reps

        @staticmethod
        def time_iterations(command, iterations):
            r, w = os.pipe()
            try:
                command = [
                    '/usr/bin/time', f'--output=/dev/fd/{w}', '--format', BenchResult.time_format,
                    '/usr/bin/xargs', '-n1', '-0',
                ] + command
                os.set_inheritable(w, True)

                result = subprocess_run(
                    command,
                    input=b'\0' * iterations,
                    stdout=subprocess.DEVNULL,
                    close_fds=False,
                )
                raw_times = list(map(float, os.read(r, 4096).decode('ascii').strip().split()))
                return result, raw_times

            finally:
                os.close(r)
                os.close(w)

elif sys.platform == 'darwin':
    class BenchResult(collections.namedtuple('BenchResult', '''
        t_sys t_user t_wall
        n_ctxswitch n_swap n_fault n_reclaim
        avg_unshared max_rss
    '''), BenchResultBase):
        _formats    = '6.2f 6.2f 6.2f  5.2f .1f .1f .1f  .0f .0f'.split()
        time_format_durations = 3 # the first 3 numbers are durations (in seconds), to be scaled by 1000 (to ms)
        time_format_per_rep = 7 # the first 7 numbers should be divided by number of reps

        _time_entries = [
            'sys',
            'user',
            'real',

            #'voluntary context switches',
            'involuntary context switches',
            'swaps',
            'page faults',
            'page reclaims',

            'average unshared data size',
            'maximum resident set size',

            #'average shared memory size',
            #'average unshared stack size',
            #'block input operations',
            #'block output operations',
            #'messages sent',
            #'messages received',
            #'signals received',
        ]

        @staticmethod
        def time_iterations(command, iterations):
            command = [
                '/usr/bin/time', '-pl',
                'sh', '-c', 'while read; do "$@"; done', '--',
            ] + command

            result = subprocess_run(
                command,
                input='\n' * iterations,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                universal_newlines=True,
            )
            #return result, [0]*len(BenchResult._time_entries)
            raw_times = dict(
                line if line[1][:1].isdigit() else (line[1], line[0])
                for line in (
                    l.strip().split(None, 1)
                    for l in result.stderr.splitlines()
                )
                if len(line) == 2
            )
            raw_times = [
                float(raw_times[f])
                for f in BenchResult._time_entries
            ]
            return result, raw_times

else:
    raise NotImplementedError(f'bench.py: Unsupported platform {sys.platform!r}!')


def shellquote(*words):
    return ' '.join(shlex.quote(c) for c in words)

def subprocess_run(command, **kwargs):
    #print(command)
    try:
        return subprocess.run(command, **kwargs)
    except:
        print(f'\nException during execution of: {shellquote(*command)}')
        raise


def bench_one(sevenzip, testfile, n_samples):
    command = [ sevenzip, '-so', 'x', testfile ]

    # warmup
    t = time.time()
    #result = subprocess_run(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
    result, raw_times = BenchResult.time_iterations(command, iterations=1)
    t = time.time() - t
    wall_time = raw_times[BenchResult._fields.index('t_wall')]
    if result.returncode != 0:
        raise SystemExit(f'\nbench.py: warmup command failed with status {result.returncode}: {shellquote(*result.args)}')
    if t > 0.05 and not (t - 0.04 < wall_time < t):
        raise SystemExit(f'\nbench.py: warmup measurement failed to measure wall time (t_wall = {wall_time}, t = {t}): {shellquote(*result.args)}')

    effective_minor_repetitions = sample_repetitions or int(math.ceil(sample_time / t))

    # actual measurement
    results = []
    for _ in range(n_samples):
        result, raw_times = BenchResult.time_iterations(command, effective_minor_repetitions)
        if result.returncode != 0:
            raise SystemExit(f'\nbench.py: benchmarking command failed with status {result.returncode}: {shellquote(*result.args)}')

        for i in range(BenchResult.time_format_durations):
            raw_times[i] *= 1000    # convert to milliseconds
        for i in range(BenchResult.time_format_per_rep):
            raw_times[i] /= effective_minor_repetitions    # calculate mean (per minor repetition)
        results.append(BenchResult(*raw_times))

    return results


def make(target):
    result = subprocess.run(
        ['make', '-s', '-j4', target],
        cwd=makeroot,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    if result.returncode != 0:
        error = result.stderr.rstrip().rsplit('\n')[-1]
        raise SystemExit(f'bench.py: make {target} failed with status {result.returncode}:\n{error}')


def update_makefile(**settings):
    with open(makefile, 'r') as f:
        text = f.read()
    for key, value in sorted(settings.items()):
        text = re.sub(fr'^{re.escape(key)}=.*', f'{key}={value}', text, flags=re.M)
    with open(makefile, 'w') as f:
        f.write(text)


class PrebuiltVariant(collections.namedtuple('PrebuiltVariant', 'name path')):
    @property
    def id(self):
        return urllib.parse.quote(self.name)

    @property
    def name30(self):
        """ __str__, except padded/truncated to 30 characters. """
        limit = 30
        return self.name.ljust(limit) if len(self.name) <= limit else (self.name[:limit - 1] + '\u2026')

    def build(self):
        raise NotImplementedError(f'{self.path!r} should be prebuilt')

    def __str__(self):
        return self.name


class BuildVariant(collections.namedtuple('BuildVariant', 'cc cxx cc_version opt')):
    @property
    def id(self):
        return '+'.join(urllib.parse.quote(v) for v in self)

    @property
    def name30(self):
        """ __str__, except padded/truncated to 30 characters. """
        limit = 30 - 1 - len(self.cc)
        opt = self.opt.ljust(limit) if len(self.opt) <= limit else (self.opt[:limit - 1] + '\u2026')
        return f'{self.cc} {opt}'

    @property
    def path(self):
        return f'{buildvariants_path}/7za+{self.id}'

    def build(self):
        opt = self.opt
        if self.cc == 'clang':
            # HACK: needed to compile 7za under clang 6
            opt += ' -Wno-c++11-narrowing'
            # Ubuntu 18.04 HACK for 'clang' using wrong linker path when gcc-8 is installed.
            #os.system('sudo chmod a-rx /usr/lib/gcc/x86_64-linux-gnu/8')

        update_makefile(CC=self.cc, CXX=self.cxx, OPTFLAGS=opt)

        make('clean')
        t = time.time()
        make(maketarget)
        t = time.time() - t
        subprocess.run(['strip', '-s', '-o', self.path, maketarget_path], check=True)

        #os.system('sudo chmod a+rx /usr/lib/gcc/x86_64-linux-gnu/8')

        size = os.path.getsize(self.path) / 1000
        with open(self.path, 'rb') as f:
            zipped_size = len(zlib.compress(f.read(), 9)) / 1000 # approximate (not counting .zip framing)
        return f'{t:3.0f} s  {size:4.0f} kB  {zipped_size:4.0f} zipped'

    def __str__(self):
        return f'{self.cc} ({self.cc_version}) {self.opt}'



def calc_t_value(v, p):
    """ Calculate t-table entry for one-sided test (approximation). """
    assert v > 0 and int(v) == v, f'invalid number of degrees of freedom {v}'
    assert 0.5 < p < 1, f'probability must be 0.5 < p < 1, was {p}'
    assert v > -math.log10(1 - p)*1.75, f'this simple t-value calculation is inaccurate for such low v ({v}) and high p ({p})'

    y = -math.log(4 * p * (1 - p))

    x = 0
    for f in (.6936233982e-12, .3657763036e-10, -.3231081277e-8, .8360937017e-7, -.104527497e-5, 0.5824238515e-5, .6841218299e-5, -.2250947176e-3, -.8364353589e-3, .03706987906, 1.570796288):
        x = y*(f + x)

    u = -math.sqrt(x)
    a = (x + 1) / 4
    b = ((5 * x + 16) * x + 3) / 96
    c = (((3 * x + 19) * x + 17) * x - 15) / 384
    d = ((((79 * x + 776) * x + 1482) * x - 1920) * x - 945) / 92160
    e = (((((27 * x + 339) * x + 930) * x - 1782) * x - 765) * x + 17955) / 368640
    return -u * (1 + (a + (b + (c + (d + e / v) / v) / v) / v) / v)


class BenchStat(collections.namedtuple('BenchStat', 'mean stddev')):
    def get_bounds(self, t_factor_div_sqrt_sample_size):
        """ Calculate (lower, upper) bounds on the true mean (with
            confidence given by t-factor).
        """
        return tuple(
            BenchResult(*(
                mean + sign * stddev * t_factor_div_sqrt_sample_size
                for mean, stddev in zip(*self)
            ))
            for sign in (-1, 1)
        )


def get_build_variants(test_compilers, compiler_versions, test_prebuilt):
    return [
        BuildVariant(cc, cxx, compiler_versions[cc], opt)
        for cc, cxx, test_options in test_compilers
        for opt in test_options
    ] + [
        PrebuiltVariant(name, path)
        for name, path in sorted(test_prebuilt.items())
    ]


TestFile = collections.namedtuple('TestFile', 'name path')
test_files = [
    TestFile(f'{os.path.basename(os.path.dirname(p))} ({friendly_size(os.path.getsize(p))})', p)
    for p in test_files
]


def measure_performance():
    # Determine compiler versions
    re_version = re.compile(r'[0-9]+(\.[0-9]+){2,}([-+~][-+~a-zA-Z0-9.]+)?')
    compiler_versions = {}
    for cc, cxx, _ in test_compilers:
        result = subprocess.run([cc, '--version'], stdout=subprocess.PIPE, check=True, universal_newlines=True)
        m = re_version.search(result.stdout)
        if not m:
            raise SystemExit(f'bench.py: could not parse {cc!r} version output:\n{result.stdout}')
        compiler_versions[cc] = m.group(0)
    compiler_versions_text = ', '.join('%s %s' % cv for cv in sorted(compiler_versions.items()))
    if compiler_versions_text:
        print(f'Using {compiler_versions_text}.')

    build_variants = get_build_variants(test_compilers, compiler_versions, test_prebuilt)

    # Compile variants
    unbuilt_variants = [b for b in build_variants if not os.path.exists(b.path)]
    n_cached = len(build_variants) - len(unbuilt_variants)
    print(f'**** Building {len(unbuilt_variants)} of {len(build_variants)} variants ({n_cached} cached)')
    os.makedirs(buildvariants_path, exist_ok=True)
    for variant in unbuilt_variants:
        build_perf = variant.build()
        print(f'{build_perf:<30}  {variant}')

    # Measure perf
    results_by_variant_and_test_name = collections.defaultdict(lambda: collections.defaultdict(list))
    combos = [(v, tf) for v in build_variants for tf in test_files]
    n = 0
    count = sample_count * len(combos)
    t = datetime.datetime.now()

    for rep in range(sample_count):
        random.shuffle(combos)
        for variant, tf in combos:
            elapsed = datetime.datetime.now() - t
            eta = 'n/a' if n == 0 else ((count - n)/n*elapsed)
            sys.stderr.write(f'\rSampling {n} of {count} ({n*100/count:.1f}%); {str(elapsed).split(".")[0]} min, ETA {str(eta).split(".")[0]} ... ')
            sys.stderr.flush()

            single = bench_one(variant.path, tf.path, n_samples=1)
            assert len(single) == 1
            results_by_variant_and_test_name[variant.id][tf.name].extend(single)
            n += 1
    sys.stderr.write(f'\r{" "*50}\r')
    sys.stderr.flush()

    if True:
        # Report individual perf results
        for variant in build_variants:
            if sample_repetitions is None:
                print(f'**** {variant}  ({sample_count} samples, repeating for at least {sample_time} seconds per sample)')
            else:
                print(f'**** {variant}  ({sample_count} samples, {sample_repetitions} repetitions per sample, {sample_count*sample_repetitions} total per test file)')

            for tf in test_files:
                results = results_by_variant_and_test_name[variant.id][tf.name]
                assert len(results) == sample_count, (variant.id, tf.name, len(results), sample_count)
                print(f'{tf.name}')
                for result in results:
                    print(f'  {result}')

    with open(raw_results_file, 'w') as f:
        json.dump({
            'compiler_versions': compiler_versions,
            'sample_count': sample_count,
            'results_by_variant_and_test_name': results_by_variant_and_test_name,
            'test_compilers': test_compilers,
            'test_prebuilt': test_prebuilt,
        }, f)


def report_performance():
    with open(raw_results_file, 'r') as f:
        cached = json.load(f)
    sample_count = cached['sample_count']
    results_by_variant_and_test_name = cached['results_by_variant_and_test_name']
    compiler_versions = cached['compiler_versions']
    test_compilers = cached['test_compilers']
    test_prebuilt = cached.get('test_prebuilt') or {}
    build_variants = get_build_variants(test_compilers, compiler_versions, test_prebuilt)

    results = {
        variant: {
            test_name: BenchStat(
                mean=BenchResult.reduce(statistics.mean, results),
                stddev=BenchResult.reduce(statistics.stdev, results), # sample stddev
            )
            for test_name, results in results_by_test_name.items()
        }
        for variant, results_by_test_name in results_by_variant_and_test_name.items()
    }

    confidence_one_sided = 1 - (1 - confidence_two_sided)/2

    t_factor = calc_t_value(p=confidence_one_sided, v=sample_count - 1)
    t_factor_div_sqrt_sample_size = t_factor/math.sqrt(sample_count)

    def calc_advantage(stat1, stat2):
        # The 'advantage' of stat1 over stat2 is the positive difference
        # between the upper bound on stat1's mean and the lower bound on
        # stat2's mean.

        m1_lower, m1_upper = stat1.get_bounds(t_factor_div_sqrt_sample_size)
        m2_lower, m2_upper = stat2.get_bounds(t_factor_div_sqrt_sample_size)

        result = []
        for i in range(len(BenchResult._fields)):
            advantage_to_1 = m1_upper[i] < 0.99 * m2_lower[i]
            advantage_to_2 = m2_upper[i] < 0.99 * m1_lower[i]
            assert not (advantage_to_1 and advantage_to_2), f'at most one can have advantage (field #{i})'
            #if i == 1:
            #    print(f'M1 {m1_lower[i]:.1f} .. {m1_upper[i]:.1f}  {advantage_to_1}')
            #    print(f'M2 {m2_lower[i]:.1f} .. {m2_upper[i]:.1f}  {advantage_to_2}')
            result.append(
                m1_upper[i] / m2_lower[i] - 1 if advantage_to_1 else # always negative
                m1_lower[i] / m2_upper[i] - 1 if advantage_to_2 else # always positive
                None
            )
        return BenchResult(*result)

    ansi_bold, ansi_end = '\x1b[1m', '\x1b[0m'
    down, up = directions = (f'\x1b[92m▾{ansi_end}', f'\x1b[91m▴{ansi_end}')
    def render_direction(advantage):
        return directions[advantage>0]

    print(f'\nShowing results, including comparison to other variants ({down} is better, {confidence_two_sided*100:.0f}% confidence).\n')
    for variant1 in build_variants:
        for test_name, stat1 in sorted(results[variant1.id].items()):
            print(f'{ansi_bold}**** {variant1}{ansi_end}  {test_name}')
            print(f'mean  {stat1.mean}')
            print(f'sd    {stat1.stddev}')
            print()

            advantages = []
            for variant2 in build_variants:
                if variant2 == variant1: continue

                stat2 = results[variant2.id][test_name]
                advantages.append((variant2, calc_advantage(stat1, stat2)))

            if advantages:
                fields = [
                    field
                    for i, field in enumerate(BenchResult._fields)
                    if any(a[1][i] for a in advantages)
                ]
                #print(f'Better than {len(advantages)} variant(s) ({confidence_two_sided*100:.0f}% confidence):')
                for variant2, advantage in advantages:
                    advantage = '  '.join(
                        f'{field}:{100*abs(adv):3.0f}%{render_direction(adv)}'
                        if adv is not None
                        else ' ' * (len(field) + 6)
                        for field, adv in zip(BenchResult._fields, advantage)
                        if field in fields
                    )
                    print(f'{variant2.name30} {advantage}')
            print()


if __name__ == '__main__':
    if os.path.exists(raw_results_file):
        print(f'**** Using cached measurements from {raw_results_file}')
    else:
        measure_performance()

    report_performance()
