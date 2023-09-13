#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Links the path of python3 include directory to a buck2 out directory.

set -e

# "-I/path/to/include -I/path/to/include" -> "/path/to/include"
LIB=$(python3-config --includes | cut -d' ' -f1 | sed -r 's/^.{2}//')

ln -s "$LIB" "$1"
