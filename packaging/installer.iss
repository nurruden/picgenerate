; 需要先安装 Inno Setup 6，并已经生成 dist\generatepic\
; 编译：iscc packaging\installer.iss

#define AppName "generatepic"
#define AppVersion "0.1.0"

[Setup]
AppId={{A7B1E4C2-6D58-4F0A-9C31-8E2F0A1B7D44}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher=generatepic
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
OutputDir=..\dist
OutputBaseFilename=generatepic-setup
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
UninstallDisplayIcon={app}\generatepic.exe

[Files]
Source: "..\dist\generatepic\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\generatepic"; Filename: "{app}\generatepic.exe"
Name: "{autodesktop}\generatepic"; Filename: "{app}\generatepic.exe"

[Run]
Filename: "{app}\generatepic.exe"; Description: "启动 generatepic"; Flags: nowait postinstall skipifsilent
