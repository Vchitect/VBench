@echo off
setlocal enabledelayedexpansion

set ALL_DIMS=subject_consistency background_consistency aesthetic_quality imaging_quality temporal_style overall_consistency human_action temporal_flickering motion_smoothness dynamic_degree
:: object_class multiple_objects needs perceptron2
:: appearance_style scene spatial_relationship color not supported for custom_input

:: Parse --videos_path and --output_path from args
set VIDEOS_PATH=
set OUTPUT_PATH=./evaluation_results
set _NEXT=
for %%A in (%*) do (
    if defined _NEXT (
        if "!_NEXT!"=="videos_path" set VIDEOS_PATH=%%~A
        if "!_NEXT!"=="output_path" set OUTPUT_PATH=%%~A
        set _NEXT=
    ) else (
        if "%%~A"=="--videos_path" set _NEXT=videos_path
        if "%%~A"=="--output_path" set _NEXT=output_path
    )
)

:: Get basename of videos_path for CSV prefix
set FOLDER_NAME=results
if defined VIDEOS_PATH (
    for %%F in ("!VIDEOS_PATH!") do set FOLDER_NAME=%%~nxF
)

:: If first arg doesn't start with --, treat it as a shorthand dimension name
set FIRST_ARG=%~1
if defined FIRST_ARG (
    set FIRST_TWO=!FIRST_ARG:~0,2!
    if not "!FIRST_TWO!"=="--" (
        set DIM=!FIRST_ARG!
        shift
        set REST=
        :argloop
        if "%~1"=="" goto argdone
        set REST=!REST! %1
        shift
        goto argloop
        :argdone
        python evaluate.py !REST! --dimension !DIM! --mode custom_input
        goto :postprocess
    )
)

:: Inject --dimension ALL_DIMS if not already specified
echo.%* | findstr /i /c:"--dimension" >nul
if errorlevel 1 (
    python evaluate.py %* --dimension %ALL_DIMS% --mode custom_input
) else (
    python evaluate.py %* --mode custom_input
)

:postprocess
python write_csvs.py --output_path "!OUTPUT_PATH!" --prefix "!FOLDER_NAME!"

endlocal
