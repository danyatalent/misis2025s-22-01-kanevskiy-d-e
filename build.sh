#!/bin/bash

read -p "Enter full path to the CMake toolchain file: " TOOLCHAIN_PATH

if [ ! -f "$TOOLCHAIN_PATH" ]; then
  echo "Toolchain file not found: $TOOLCHAIN_PATH"
  exit 1
fi

cmake -S . -B ./build -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN_PATH"
cmake --build ./build
