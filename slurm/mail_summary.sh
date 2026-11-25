#!/bin/bash
# One summary mail for a finished chain; run it from an afterany job.
# Usage: mail_summary.sh <to> <subject> <job id>...
# Per job: the sacct row, the key result lines of its .out, and for any .err holding a traceback
# its last 20 lines, so a failure is readable from the mail without logging in.
to=${1:?to} subject=${2:?subject}
shift 2
repo="${IRIS_REPO:-$HOME/Iris}"
ids=$(IFS=,; echo "$*")
{
    sacct -X -j "$ids" -o JobID,JobName%30,State,Elapsed,ExitCode
    for j in "$@"; do
        for f in "$repo"/*-"$j".out "$repo"/*-"$j"_*.out; do
            [[ -f "$f" ]] || continue
            echo "--- $f"
            grep -h -E '^\[(preflight|control|row|use_average|reproduced|gt-isolation|selected|verdict|test-official)\]|^\[smoke\] PASS|affine_invariant_log_space: abs_rel' "$f" | sort -u || true
        done
        for f in "$repo"/*-"$j".err "$repo"/*-"$j"_*.err; do
            [[ -f "$f" ]] && grep -q -E 'Traceback|Error|error:' "$f" || continue
            echo "--- $f (last 20 lines)"
            tail -c 4000 "$f" | tr '\r' '\n' | tail -n 20
        done
    done
} 2>&1 | mail -s "$subject" "$to"
