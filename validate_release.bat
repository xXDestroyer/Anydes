@echo off
setlocal
cd /d "%~dp0"
echo Anydes source release preflight - this is not the full model distribution.
python -B tools\check_source.py
echo For full runtime prerequisites, read docs\PORTABILITY.md and docs\MODELS.md.
endlocal
