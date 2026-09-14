# PyInstaller spec: windowed one-directory build (spec section 11).
import os
from PyInstaller.utils.hooks import collect_data_files

block_cipher = None
here = os.path.dirname(os.path.abspath(SPEC))
assets = os.path.join(here, "..", "src", "yololabeler", "assets")

datas = collect_data_files("customtkinter")
datas += [(os.path.join(assets, f), os.path.join("yololabeler", "assets"))
          for f in os.listdir(assets)]

a = Analysis([os.path.join(here, "launch.py")], pathex=[os.path.join(here, "..", "src")],
             binaries=[], datas=datas, hiddenimports=["PIL._tkinter_finder"],
             hookspath=[], runtime_hooks=[], excludes=[], cipher=block_cipher)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="YoloLabeler",
          debug=False, console=False,
          icon=os.path.join(assets, "app_icon.ico"))
coll = COLLECT(exe, a.binaries, a.zipfiles, a.datas, name="YoloLabeler")
