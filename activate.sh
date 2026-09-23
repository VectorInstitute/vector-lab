#!/usr/bin/env bash
# Source this file from ~/.bashrc to make vector-lab available in every shell:
#   source /absolute/path/to/vector-lab/activate.sh

VECTOR_LAB_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
export VECTOR_LAB_HOME

case ":${PATH}:" in
    *":${VECTOR_LAB_HOME}/bin:"*) ;;
    *) export PATH="${VECTOR_LAB_HOME}/bin:${PATH}" ;;
esac
