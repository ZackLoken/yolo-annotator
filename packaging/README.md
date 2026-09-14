# Building the Windows app

    pip install pyinstaller
    pyinstaller packaging/yololabeler.spec --noconfirm --distpath packaging/dist --workpath packaging/build

Result: `packaging/dist/YoloLabeler/YoloLabeler.exe`. Zip the folder and send it.
The executable is unsigned; the first run shows a SmartScreen prompt. Choose
"More info" then "Run anyway".

Test on a machine without conda or Python: open a folder, import predictions,
accept one, quit, and confirm the label file changed.
