@echo off

set spirv_dir=.
set glslang=glslc.exe

for %%f in (*.rgen) do (
	echo Compiling %%f
	%glslang% -o %spirv_dir%\%%~nf.rgen.spv %%f || (goto :error)
)

for %%f in (*.rmiss) do (
	echo Compiling %%f
    %glslang% -o %spirv_dir%\%%~nf.rmiss.spv %%f || (goto :error)
)

for %%f in (*.rchit) do (
	echo Compiling %%f
    %glslang% -o %spirv_dir%\%%~nf.rchit.spv %%f || (goto :error)
)

for %%f in (*.rahit) do (
	echo Compiling %%f
    %glslang% -o %spirv_dir%\%%~nf.rahit.spv %%f || (goto :error)
)

for %%f in (*.vert) do (
	echo Compiling %%f
    %glslang% -o %spirv_dir%\%%~nf.vert.spv %%f || (goto :error)
)

for %%f in (*.frag) do (
	echo Compiling %%f
    %glslang% -o %spirv_dir%\%%~nf.frag.spv %%f || (goto :error)
)

goto :eof

:error
	echo Error
	pause