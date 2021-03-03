# -*- mode: python ; coding: utf-8 -*-

block_cipher = None


a = Analysis(['new_window.py',r'D:\PycharmProjects\Mask Detection\UI\new_ui.py',r'D:\PycharmProjects\Mask Detection\UI\ui_utils.py',r'D:\PycharmProjects\Mask Detection\yolo_net\Class_Yolo.py',r'D:\PycharmProjects\Mask Detection\yolo_net\CSPDarknet53.py',r'D:\PycharmProjects\Mask Detection\yolo_net\Yolo_Net.py',r'D:\PycharmProjects\Mask Detection\utils\utils.py',r'D:\PycharmProjects\Mask Detection\predict\image_predict.py',r'D:\PycharmProjects\Mask Detection\predict\video_predict.py'],
             pathex=['D:\\PycharmProjects\\Mask Detection\\UI'],
             binaries=[],
             datas=[],
             hiddenimports=[],
             hookspath=[],
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
          [],
          exclude_binaries=True,
          name='new_window',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=True )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               upx_exclude=[],
               name='new_window')
