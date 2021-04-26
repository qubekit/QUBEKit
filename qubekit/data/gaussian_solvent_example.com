%Mem=1GB
%NProcShared=1
%Chk=lig
# EmpiricalDispersion=GD3BJ b3lyp/6-311G  SCF=(XQC, MaxConventionalCycles=300) nosymm SP SCRF=(IPCM, Read) density=current OUTPUT=WFX

gaussian job

0 1
O   0.0610000  0.3890000  0.0590000
H   0.7240000 -0.3120000 -0.0380000
H  -0.7850000 -0.0770000 -0.0210000

4.0 0.0004
gaussian.wfx


