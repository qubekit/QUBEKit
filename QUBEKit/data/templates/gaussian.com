%Mem={{ memory }}GB
%NProcShared={{ threads }}
%Chk=lig
# {{ theory }}/{{ basis }}  SCF=(XQC, MaxConventionalCycles={{ scf_maxiter }}) nosymm {{ driver }}

{{ title }}

{{ charge }} {{ multiplicity }}
{%- for element, coords in data %}
{{ element }}  {{ '{: .10f}'.format(coords[0]) }} {{ '{: .10f}'.format(coords[1]) }} {{ '{: .10f}'.format(coords[2]) }}
{%- endfor %}

