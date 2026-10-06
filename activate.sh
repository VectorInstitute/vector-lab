#!/usr/bin/env bash
# Source this file from ~/.bashrc to make vector-sim available in every shell:
#   source /absolute/path/to/vector-sim/activate.sh

VECTOR_SIM_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
export VECTOR_SIM_HOME

case ":${PATH}:" in
    *":${VECTOR_SIM_HOME}/bin:"*) ;;
    *) export PATH="${VECTOR_SIM_HOME}/bin:${PATH}" ;;
esac
