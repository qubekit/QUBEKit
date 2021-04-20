%Mem=1GB
%NProcShared=1
%Chk=lig
# EmpiricalDispersion=GD3BJ b3lyp/6-311G  SCF=(XQC, MaxConventionalCycles=300) nosymm SP SCRF=(IPCM, Read) density=current OUTPUT=WFX

gaussian job

0 1
O   0.0610000002  0.3890000024  0.0590000022
H   0.7240000003 -0.3120000022 -0.0379999993
H  -0.7850000005 -0.0770000002 -0.0209999977

4.0 0.0004
gaussian.wfx


