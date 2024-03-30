# -*- mode: python ; coding: utf-8 -*-
# Activate environment, then type "pyinstaller crt_exe_gui.spec" in command line
# to generate .exe file for gui


block_cipher = None


a = Analysis(['main_gui.py'],
             pathex=[],
             binaries=[],
             datas=[('env/Lib/site-packages/mediapipe/modules', 'mediapipe/modules'),
			 ('models', 'models'),
			 ('docs', 'docs'),
             ('other_files', 'other_files')],
             hiddenimports=[],
             hookspath=[],
             hooksconfig={},
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)

exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,  
          [],
          name='Owleye',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=True,
		  icon='docs/images/logo.ico',
          disable_windowed_traceback=False,
          target_arch=None,
          codesign_identity=None,
          entitlements_file=None )
