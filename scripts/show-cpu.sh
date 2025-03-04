#!/bin/bash

set -e

cat /proc/cpuinfo | grep 'processor\|cpu MHz' | cut -d ':' -f 2 | awk '{print $1}' | xargs -n2
