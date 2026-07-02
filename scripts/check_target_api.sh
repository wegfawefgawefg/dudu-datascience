#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
manifest="$repo_root/spec/target_api/manifest.tsv"
dudu_bin="${DUDU_BIN:-dudu}"

cd "$repo_root"

graduated=0
pending=0

while IFS=$'\t' read -r file status evidence; do
    [[ -z "${file:-}" || "${file:0:1}" == "#" ]] && continue

    case "$status" in
        graduated)
            if [[ ! -f "spec/target_api/$file" ]]; then
                printf '[target-api] missing spec file: %s\n' "$file" >&2
                exit 1
            fi
            if [[ -z "${evidence:-}" ]]; then
                printf '[target-api] graduated spec has no runnable target: %s\n' "$file" >&2
                exit 1
            fi
            printf '[target-api] run %s for %s\n' "$evidence" "$file" >&2
            "$dudu_bin" run "$evidence" >/tmp/dudu-datascience-"$evidence".out
            graduated=$((graduated + 1))
            ;;
        pending)
            if [[ ! -f "spec/target_api/$file" ]]; then
                printf '[target-api] missing pending spec file: %s\n' "$file" >&2
                exit 1
            fi
            if [[ -z "${evidence:-}" ]]; then
                printf '[target-api] pending spec needs an explicit reason: %s\n' "$file" >&2
                exit 1
            fi
            printf '[target-api] pending %s: %s\n' "$file" "$evidence" >&2
            pending=$((pending + 1))
            ;;
        *)
            printf '[target-api] invalid status for %s: %s\n' "$file" "$status" >&2
            exit 1
            ;;
    esac
done <"$manifest"

printf '[target-api] %d graduated, %d pending\n' "$graduated" "$pending" >&2
