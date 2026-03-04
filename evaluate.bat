@echo off
setlocal enabledelayedexpansion

set ALL_DIMS=subject_consistency background_consistency aesthetic_quality imaging_quality object_class multiple_objects color spatial_relationship scene temporal_style overall_consistency human_action temporal_flickering motion_smoothness dynamic_degree appearance_style

:: Inject --dimension ALL_DIMS if not already specified
echo.%* | findstr /i "\-\-dimension" >nul
if errorlevel 1 (
    python evaluate.py %* --dimension %ALL_DIMS%
) else (
    python evaluate.py %*
)

endlocal
