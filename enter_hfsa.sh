#!/usr/bin/env bash
HFSA_ROOT="/mnt/d/code/python/HFSA"
HFSA_MAIN="$HFSA_ROOT/HFSA-main"
HFSA_ACTIVATE="$HFSA_ROOT/hfsa_env/bin/activate"

HFSA_SOURCED=0
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  HFSA_SOURCED=1
fi

hfsa_fail() {
  echo "$1" >&2
  if [[ "$HFSA_SOURCED" -eq 1 ]]; then
    return 1
  fi
  exit 1
}

hfsa_enter_current_shell() {
  cd "$HFSA_ROOT" || return 1
  # shellcheck source=/dev/null
  source "$HFSA_ACTIVATE" || return 1
  cd "$HFSA_MAIN" || return 1

  echo "HFSA environment is ready."
  echo "Directory: $(pwd)"
  echo "Python: $(command -v python)"
}

if [[ ! -f "$HFSA_ACTIVATE" ]]; then
  hfsa_fail "Cannot find virtualenv activate script: $HFSA_ACTIVATE" || return 1
fi

if [[ ! -d "$HFSA_MAIN" ]]; then
  hfsa_fail "Cannot find project directory: $HFSA_MAIN" || return 1
fi

if [[ "$HFSA_SOURCED" -eq 1 ]]; then
  hfsa_enter_current_shell
  return $?
fi

HFSA_RCFILE="$(mktemp)"
{
  printf '[[ -f ~/.bashrc ]] && source ~/.bashrc\n'
  printf 'cd %q\n' "$HFSA_ROOT"
  printf 'source %q\n' "$HFSA_ACTIVATE"
  printf 'cd %q\n' "$HFSA_MAIN"
  printf 'echo "HFSA environment is ready."\n'
  printf 'echo "Directory: $(pwd)"\n'
  printf 'echo "Python: $(command -v python)"\n'
  printf 'rm -f %q\n' "$HFSA_RCFILE"
} > "$HFSA_RCFILE"

exec bash --rcfile "$HFSA_RCFILE" -i
