@echo off
setlocal

echo Enter full path to the CMake toolchain file:
set /p TOOLCHAIN_PATH=

if not exist "%TOOLCHAIN_PATH%" (
    echo Toolchain file not found: %TOOLCHAIN_PATH%
    exit /b 1
)

cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE="%TOOLCHAIN_PATH%"
cmake --build build

endlocal
