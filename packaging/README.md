# Building the Windows app

    pip install -e ".[packaging]"
    pyinstaller packaging/yololabeler.spec --noconfirm --distpath packaging/dist --workpath packaging/build

Result: `packaging/dist/YoloLabeler/YoloLabeler.exe`. Zip the folder and send it.
The executable is unsigned; the first run shows a SmartScreen prompt. Choose
"More info" then "Run anyway".

Test on a machine without conda or Python: open a folder that already holds a
`predictions/` tree, accept one prediction, quit, and confirm the label file
changed. The packaged app has no importer; predictions come from the
`yololabeler-import` command, which ships with the pip install rather than with
this build.
