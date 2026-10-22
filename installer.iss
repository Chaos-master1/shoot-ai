# Inno Setup Script for Kawkab AI
# Creates a professional Windows installer

#define MyAppName "Kawkab AI"
#define MyAppVersion "0.1.0"
#define MyAppPublisher "Kawkab AI"
#define MyAppURL "https://github.com/wizard404error/shoot-ai"
#define MyAppExeName "KawkabAI.exe"

[Setup]
; NOTE: AppId is a unique GUID identifying this application
AppId={{A8B9C7D6-E5F4-4A3B-9C8D-1E2F3A4B5C6D}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}/issues
AppUpdatesURL={#MyAppURL}/releases
DefaultDirName={autopf}\KawkabAI
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
LicenseFile=LICENSE
OutputDir=installer
OutputBaseFilename=KawkabAI-Setup-{#MyAppVersion}
SetupIconFile=assets\icon.ico
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=admin
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
UninstallDisplayIcon={app}\{#MyAppExeName}
UninstallDisplayName={#MyAppName}

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "arabic"; MessagesFile: "compiler:Languages\Arabic.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; PyInstaller output folder
Source: "dist\KawkabAI\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
; Knowledge base
Source: "src\kawkab\knowledge\*"; DestDir: "{app}\kawkab\knowledge"; Flags: ignoreversion recursesubdirs createallsubdirs
; Web frontend
Source: "src\kawkab\web\*"; DestDir: "{app}\kawkab\web"; Flags: ignoreversion recursesubdirs createallsubdirs
; License
Source: "LICENSE"; DestDir: "{app}"; Flags: ignoreversion
; README
Source: "README.md"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\{cm:UninstallProgram,{#MyAppName}}"; Filename: "{uninstallexe}"
Name: "{commondesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#MyAppName}}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{app}\kawkab"
Type: filesandordirs; Name: "{app}\models"
