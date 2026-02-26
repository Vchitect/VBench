@echo off
setlocal enabledelayedexpansion

if "%~1"=="" goto :original_mode

:: Pass all arguments directly to evaluate.py
python evaluate.py %*
goto :eof

:original_mode
set BASE_PATH=.\vbench_videos\

set MODELS=lavie modelscope videocrafter cogvideo

set DIM[0]=subject_consistency   & set FOL[0]=subject_consistency
set DIM[1]=background_consistency & set FOL[1]=scene
set DIM[2]=aesthetic_quality      & set FOL[2]=overall_consistency
set DIM[3]=imaging_quality        & set FOL[3]=overall_consistency
set DIM[4]=object_class           & set FOL[4]=object_class
set DIM[5]=multiple_objects       & set FOL[5]=multiple_objects
set DIM[6]=color                  & set FOL[6]=color
set DIM[7]=spatial_relationship   & set FOL[7]=spatial_relationship
set DIM[8]=scene                  & set FOL[8]=scene
set DIM[9]=temporal_style         & set FOL[9]=temporal_style
set DIM[10]=overall_consistency   & set FOL[10]=overall_consistency
set DIM[11]=human_action          & set FOL[11]=human_action
set DIM[12]=temporal_flickering   & set FOL[12]=temporal_flickering
set DIM[13]=motion_smoothness     & set FOL[13]=subject_consistency
set DIM[14]=dynamic_degree        & set FOL[14]=subject_consistency
set DIM[15]=appearance_style      & set FOL[15]=appearance_style

for %%M in (%MODELS%) do (
    for /L %%I in (0,1,15) do (
        set VIDEOS_PATH=%BASE_PATH%%%M\!FOL[%%I]!
        echo !DIM[%%I]! !VIDEOS_PATH!
        python evaluate.py --videos_path "!VIDEOS_PATH!" --dimension !DIM[%%I]!
    )
)

endlocal
