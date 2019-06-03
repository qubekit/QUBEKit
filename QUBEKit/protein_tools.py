#!/usr/bin/env python3

import numpy as np

import simtk.openmm as mm


def apply_opls_combo(system, switching_distance=None):
    """Apply the opls combination rules to a OpenMM system and return the new system."""

    # Get the system information from the openmm system
    forces = {system.getForce(index).__class__.__name__: system.getForce(index) for index in
              range(system.getNumForces())}
    # Use the nondonded_force to get the same rules
    nonbonded_force = forces['NonbondedForce']
    lorentz = mm.CustomNonbondedForce(
        'epsilon*((sigma/r)^12-(sigma/r)^6); sigma=sqrt(sigma1*sigma2); epsilon=sqrt(epsilon1*epsilon2)*4.0')
    lorentz.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic)
    lorentz.addPerParticleParameter('sigma')
    lorentz.addPerParticleParameter('epsilon')
    lorentz.setCutoffDistance(nonbonded_force.getCutoffDistance())
    lorentz.setUseLongRangeCorrection(True)
    if switching_distance:
        lorentz.setUseSwitchingFunction(True)
        lorentz.setSwitchingDistance(switching_distance)
    system.addForce(lorentz)

    l_j_set = {}
    # For each particle, calculate the combination list again
    for index in range(nonbonded_force.getNumParticles()):
        charge, sigma, epsilon = nonbonded_force.getParticleParameters(index)
        l_j_set[index] = (sigma, epsilon, charge)
        lorentz.addParticle([sigma, epsilon])
        nonbonded_force.setParticleParameters(index, charge, 0, 0)

    for i in range(nonbonded_force.getNumExceptions()):
        (p1, p2, q, sig, eps) = nonbonded_force.getExceptionParameters(i)
        # ALL THE 12,13 and 14 interactions are EXCLUDED FROM CUSTOM NONBONDED FORCE
        lorentz.addExclusion(p1, p2)
        if eps._value != 0.0:
            charge = 0.5 * (l_j_set[p1][2] * l_j_set[p2][2])
            sig14 = np.sqrt(l_j_set[p1][0] * l_j_set[p2][0])
            nonbonded_force.setExceptionParameters(i, p1, p2, charge, sig14, eps)

    return system


def get_water(water=None):
    """Print an opls compatible water model to be used with the qube force field."""

    tip3p = '''<ForceField>
 <AtomTypes>
  <Type name="tip3p-O" class="OW" element="O" mass="15.99943"/>
  <Type name="tip3p-H" class="HW" element="H" mass="1.007947"/>
 </AtomTypes>
 <Residues>
  <Residue name="HOH">
   <Atom name="O" type="tip3p-O"/>
   <Atom name="H1" type="tip3p-H"/>
   <Atom name="H2" type="tip3p-H"/>
   <Bond atomName1="O" atomName2="H1"/>
   <Bond atomName1="O" atomName2="H2"/>
  </Residue>
 </Residues>
 <HarmonicBondForce>
  <Bond class1="OW" class2="HW" length="0.09572" k="462750.4"/>
 </HarmonicBondForce>
 <HarmonicAngleForce>
  <Angle class1="HW" class2="OW" class3="HW" angle="1.82421813418" k="836.8"/>
 </HarmonicAngleForce>
 <NonbondedForce coulomb14scale="0.5" lj14scale="0.5">
  <Atom type="tip3p-O" charge="-0.834" sigma="0.31507524065751241" epsilon="0.635968"/>
  <Atom type="tip3p-H" charge="0.417" sigma="1" epsilon="0"/>
 </NonbondedForce>
</ForceField>'''

    spce = '''<ForceField>
 <AtomTypes>
  <Type name="spce-O" class="OW" element="O" mass="15.99943"/>
  <Type name="spce-H" class="HW" element="H" mass="1.007947"/>
 </AtomTypes>
 <Residues>
  <Residue name="HOH">
   <Atom name="O" type="spce-O"/>
   <Atom name="H1" type="spce-H"/>
   <Atom name="H2" type="spce-H"/>
   <Bond atomName1="O" atomName2="H1"/>
   <Bond atomName1="O" atomName2="H2"/>
  </Residue>
 </Residues>
 <HarmonicBondForce>
  <Bond class1="OW" class2="HW" length="0.1" k="462750.4"/>
 </HarmonicBondForce>
 <HarmonicAngleForce>
  <Angle class1="HW" class2="OW" class3="HW" angle="1.91061193216" k="836.8"/>
 </HarmonicAngleForce>
 <NonbondedForce coulomb14scale="0.5" lj14scale="0.5">
  <Atom type="spce-O" charge="-0.8476" sigma="0.31657195050398818" epsilon="0.6497752"/>
  <Atom type="spce-H" charge="0.4238" sigma="1" epsilon="0"/>
 </NonbondedForce>
</ForceField>'''

    tip4pew = '''<ForceField>
 <AtomTypes>
  <Type name="tip4pew-O" class="OW" element="O" mass="15.99943"/>
  <Type name="tip4pew-H" class="HW" element="H" mass="1.007947"/>
  <Type name="tip4pew-M" class="MW" mass="0"/>
 </AtomTypes>
 <Residues>
  <Residue name="HOH">
   <Atom name="O" type="tip4pew-O"/>
   <Atom name="H1" type="tip4pew-H"/>
   <Atom name="H2" type="tip4pew-H"/>
   <Atom name="M" type="tip4pew-M"/>
   <VirtualSite type="average3" siteName="M" atomName1="O" atomName2="H1" atomName3="H2" weight1="0.786646558" weight2="0.106676721" weight3="0.106676721"/>
   <Bond atomName1="O" atomName2="H1"/>
   <Bond atomName1="O" atomName2="H2"/>
  </Residue>
 </Residues>
 <HarmonicBondForce>
  <Bond class1="OW" class2="HW" length="0.09572" k="462750.4"/>
 </HarmonicBondForce>
 <HarmonicAngleForce>
  <Angle class1="HW" class2="OW" class3="HW" angle="1.82421813418" k="836.8"/>
 </HarmonicAngleForce>
 <NonbondedForce coulomb14scale="0.5" lj14scale="0.5">
  <Atom type="tip4pew-O" charge="0" sigma="0.316435" epsilon="0.680946"/>
  <Atom type="tip4pew-H" charge="0.52422" sigma="1" epsilon="0"/>
  <Atom type="tip4pew-M" charge="-1.04844" sigma="1" epsilon="0"/>
 </NonbondedForce>
</ForceField>'''

    tip5p = '''<ForceField>
 <AtomTypes>
  <Type name="tip5p-O" class="OW" element="O" mass="15.99943"/>
  <Type name="tip5p-H" class="HW" element="H" mass="1.007947"/>
  <Type name="tip5p-M" class="MW" mass="0"/>
 </AtomTypes>
 <Residues>
  <Residue name="HOH">
   <Atom name="O" type="tip5p-O"/>
   <Atom name="H1" type="tip5p-H"/>
   <Atom name="H2" type="tip5p-H"/>
   <Atom name="M1" type="tip5p-M"/>
   <Atom name="M2" type="tip5p-M"/>
   <VirtualSite type="outOfPlane" siteName="M1" atomName1="O" atomName2="H1" atomName3="H2" weight12="-0.34490826" weight13="-0.34490826" weightCross="-6.4437903"/>
   <VirtualSite type="outOfPlane" siteName="M2" atomName1="O" atomName2="H1" atomName3="H2" weight12="-0.34490826" weight13="-0.34490826" weightCross="6.4437903"/>
   <Bond atomName1="O" atomName2="H1"/>
   <Bond atomName1="O" atomName2="H2"/>
  </Residue>
 </Residues>
 <HarmonicBondForce>
  <Bond class1="OW" class2="HW" length="0.09572" k="462750.4"/>
 </HarmonicBondForce>
 <HarmonicAngleForce>
  <Angle class1="HW" class2="OW" class3="HW" angle="1.82421813418" k="836.8"/>
 </HarmonicAngleForce>
 <NonbondedForce coulomb14scale="0.5" lj14scale="0.5">
  <Atom type="tip5p-O" charge="0" sigma="0.312" epsilon="0.66944"/>
  <Atom type="tip5p-H" charge="0.241" sigma="1" epsilon="0"/>
  <Atom type="tip5p-M" charge="-0.241" sigma="1" epsilon="0"/>
 </NonbondedForce>
</ForceField>'''

    tip3pfb = '''<ForceField>
 <Info>
  <DateGenerated>2014-05-28</DateGenerated>
  <Reference>Lee-Ping Wang, Todd J. Martinez and Vijay S. Pande. Building force fields - an automatic, systematic and reproducible approach.  Journal of Physical Chemistry Letters, 2014, 5, pp 1885-1891.  DOI:10.1021/jz500737m</Reference>
 </Info>
 <AtomTypes>
  <Type name="tip3p-fb-O" class="OW" element="O" mass="15.99943"/>
  <Type name="tip3p-fb-H" class="HW" element="H" mass="1.007947"/>
 </AtomTypes>
 <Residues>
  <Residue name="HOH">
   <Atom name="O" type="tip3p-fb-O"/>
   <Atom name="H1" type="tip3p-fb-H"/>
   <Atom name="H2" type="tip3p-fb-H"/>
   <Bond from="0" to="1"/>
   <Bond from="0" to="2"/>
  </Residue>
 </Residues>
 <HarmonicBondForce>
  <Bond class1="OW" class2="HW" length="0.101181082494" k="462750.4"/>
 </HarmonicBondForce>
 <HarmonicAngleForce>
  <Angle class1="HW" class2="OW" class3="HW" angle="1.88754640288" k="836.8"/>
 </HarmonicAngleForce>
 <NonbondedForce coulomb14scale="0.5" lj14scale="0.5">
  <Atom type="tip3p-fb-O" charge="-0.848448690103" sigma="0.317796456355" epsilon="0.652143528104" />
  <Atom type="tip3p-fb-H" charge="0.4242243450515" sigma="1" epsilon="0" />
 </NonbondedForce>
</ForceField>'''

    tip4pfb = '''<ForceField>
 <Info>
  <DateGenerated>2014-05-28</DateGenerated>
  <Reference>Lee-Ping Wang, Todd J. Martinez and Vijay S. Pande. Building force fields - an automatic, systematic and reproducible approach.  Journal of Physical Chemistry Letters, 2014, 5, pp 1885-1891.  DOI:10.1021/jz500737m</Reference>
 </Info>
 <AtomTypes>
  <Type name="tip4p-fb-O" class="OW" element="O" mass="15.99943"/>
  <Type name="tip4p-fb-H" class="HW" element="H" mass="1.007947"/>
  <Type name="tip4p-fb-M" class="MW" mass="0"/>
 </AtomTypes>
 <Residues>
  <Residue name="HOH">
   <Atom name="O" type="tip4p-fb-O"/>
   <Atom name="H1" type="tip4p-fb-H"/>
   <Atom name="H2" type="tip4p-fb-H"/>
   <Atom name="M" type="tip4p-fb-M"/>
   <VirtualSite type="average3" index="3" atom1="0" atom2="1" atom3="2" weight1="8.203146574531e-01" weight2="8.984267127345e-02" weight3="8.984267127345e-02" />
   <Bond from="0" to="1"/>
   <Bond from="0" to="2"/>
  </Residue>
 </Residues>
 <HarmonicBondForce>
  <Bond class1="OW" class2="HW" length="0.09572" k="462750.4"/>
 </HarmonicBondForce>
 <HarmonicAngleForce>
  <Angle class1="HW" class2="OW" class3="HW" angle="1.82421813418" k="836.8"/>
 </HarmonicAngleForce>
 <NonbondedForce coulomb14scale="0.5" lj14scale="0.5">
  <Atom type="tip4p-fb-O" charge="0" sigma="3.165552430462e-01" epsilon="7.492790213533e-01" />
  <Atom type="tip4p-fb-H" charge="5.258681106763e-01" sigma="1" epsilon="0" />
  <Atom type="tip4p-fb-M" charge="-1.0517362213526e+00" sigma="1" epsilon="0" />
 </NonbondedForce>
</ForceField>'''

    tip4pd = '''<ForceField>
 <Info>
  <Reference>Water Dispersion Interactions Strongly Influence Simulated Structural Properties of Disordered Protein States
Stefano Piana, Alexander G. Donchev, Paul Robustelli, and David E. Shaw
The Journal of Physical Chemistry B 2015 119 (16), 5113-5123
DOI: 10.1021/jp508971m </Reference>
 </Info>
 <AtomTypes>
  <Type name="tip4pew-O" class="OW" element="O" mass="15.99943"/>
  <Type name="tip4pew-H" class="HW" element="H" mass="1.007947"/>
  <Type name="tip4pew-M" class="MW" mass="0"/>
 </AtomTypes>
 <Residues>
  <Residue name="HOH">
   <Atom name="O" type="tip4pew-O"/>
   <Atom name="H1" type="tip4pew-H"/>
   <Atom name="H2" type="tip4pew-H"/>
   <Atom name="M" type="tip4pew-M"/>
   <VirtualSite type="average3" siteName="M" atomName1="O" atomName2="H1" atomName3="H2" weight1="0.786646558" weight2="0.106676721" weight3="0.106676721"/>
   <Bond atomName1="O" atomName2="H1"/>
   <Bond atomName1="O" atomName2="H2"/>
  </Residue>
 </Residues>
 <HarmonicBondForce>
  <Bond class1="OW" class2="HW" length="0.09572" k="462750.4"/>
 </HarmonicBondForce>
 <HarmonicAngleForce>
  <Angle class1="HW" class2="OW" class3="HW" angle="1.82421813418" k="836.8"/>
 </HarmonicAngleForce>
 <NonbondedForce coulomb14scale="0.5" lj14scale="0.5">
  <Atom type="tip4pew-O" charge="0" sigma="0.316435" epsilon="0.936554"/>
  <Atom type="tip4pew-H" charge="0.58" sigma="1" epsilon="0"/>
  <Atom type="tip4pew-M" charge="-1.16" sigma="1" epsilon="0"/>
 </NonbondedForce>
</ForceField>'''

    opc = '''<ForceField>
 <Info>
  <Reference>Water Dispersion Interactions Strongly Influence Simulated Structural Properties of Disordered Protein States
Stefano Piana, Alexander G. Donchev, Paul Robustelli, and David E. Shaw
The Journal of Physical Chemistry B 2015 119 (16), 5113-5123
DOI: 10.1021/jp508971m</Reference>
 </Info>
 <AtomTypes>
  <Type name="opc-O" class="OW" element="O" mass="15.99943"/>
  <Type name="opc-H" class="HW" element="H" mass="1.007947"/>
  <Type name="opc-M" class="MW" mass="0"/>
 </AtomTypes>
 <Residues>
  <Residue name="HOH">
   <Atom name="O" type="opc-O"/>
   <Atom name="H1" type="opc-H"/>
   <Atom name="H2" type="opc-H"/>
   <Atom name="M" type="opc-M"/>
   <VirtualSite type="localCoords" index="3" atom1="0" atom2="1" atom3="2" wo1="1.0" wo2="0.0" wo3="0.0" wx1="-1.0" wx2="1.0" wx3="0.0" wy1="-1.0" wy2="0.0" wy3="1.0" p1="0.00986" p2="0.01253" p3="0.000"/>
   <Bond atomName1="O" atomName2="H1"/>
   <Bond atomName1="O" atomName2="H2"/>
  </Residue>
 </Residues>
 <HarmonicBondForce>
  <Bond class1="OW" class2="HW" length="0.08724" k="462750.4"/>
 </HarmonicBondForce>
 <HarmonicAngleForce>
  <Angle class1="HW" class2="OW" class3="HW" angle="1.808161105" k="836.8"/>
 </HarmonicAngleForce>
 <NonbondedForce coulomb14scale="0.5" lj14scale="0.5">
  <Atom type="opc-O" charge="0" sigma="0.316655" epsilon="0.89036"/>
  <Atom type="opc-H" charge="0.6791" sigma="1" epsilon="0"/>
  <Atom type="opc-M" charge="-1.3582" sigma="1" epsilon="0"/>
 </NonbondedForce>
</ForceField>'''

    water_dict = {
        'tip3p': tip3p, 'spce': spce, 'tip3pfb': tip3pfb,   # 3 particle waters
        'tip4p': tip4pew, 'tip4pew': tip4pew, 'tip4p-d': tip4pd, 'tip4pfb': tip4pfb, 'opc': opc,  # 4 particle waters
        'tip5p': tip5p  # 5 particle waters
    }

    if not water or water == 'help':
        print('Please enter a water model from the current options:\nOpenMM standard '
              'models:\ntip3p\ntip4pew\ntip5p\nspce\nForcebalance models:\ntip3pfb\ntip4pfb\nExtras:\ntip4p-d\nopc')
    else:
        with open(f'QUBE_{water}.xml', 'w+') as xml:
            xml.write(water_dict[water])


def pdb_reformat(reference, target):
    """Rewrite a QUBEKit system pdb traj file to have the correct atom and residue names."""

    from QUBEKit.ligand import Protein

    # load the original file as a reference
    pro = Protein(reference)
    print(pro.pdb_names)

    # load the target file now
    with open(target, 'r') as traj:
        lines = traj.readlines()

    # now loop through the target file and replace the atom and residue names
    PRO = False
    i = 0
    new_traj = open('QUBE_traj.pdb', 'w+')
    for line in lines:
        # Find where the protein starts
        if 'MODEL' in line:
            PRO = True
            new_traj.write(line)
            i = 1
            continue
        elif 'TER' in line:
            PRO = False
            if 'QUP' in line:
                new_traj.write(f'{line[:16]} {pro.residues[-1]:4}{line[21:]}')
            else:
                new_traj.write(line)
            continue
        # If we are in the protein re-write the protein file
        if PRO:
            if len(pro.pdb_names[i-1]) <= 3:
                new_traj.write(f'ATOM   {i:4}  {pro.pdb_names[i-1]:3} {pro.Residues[i-1]:4}{line[21:]}')
            elif len(pro.pdb_names[i-1]) == 4:
                new_traj.write(f'ATOM   {i:4} {pro.pdb_names[i-1]:4} {pro.Residues[i-1]:4}{line[21:]}')
            i += 1
        else:
            new_traj.write(line)


def qube_general():
    """Write out the qube general force field to be used in parametersiation."""

    with open('QUBE_general_pi.xml', 'w+') as qube:
        qube.write('''<ForceField>
  <Info>
    <DateGenerated>2019-02-14-20190321-correctedHIS,THR,LYS,ASP issues, PRO-amino are still opls </DateGenerated>
    <Reference>modified using amber forcefield template +amber charges + opls atomtype+modified seminario parameters</Reference>
  </Info>
  <AtomTypes>
    <!-- from excel script -->
    <Type class="C" element="C" mass="12.011" name="C235"/>
    <Type class="C" element="C" mass="12.011" name="C267"/>
    <Type class="C" element="C" mass="12.011" name="C271"/>
    <Type class="CA" element="C" mass="12.011" name="C145"/>
    <Type class="CA" element="C" mass="12.011" name="C166"/>
    <Type class="CA" element="C" mass="12.011" name="C302"/>
    <Type class="CB" element="C" mass="12.011" name="C501"/>
    <Type class="CN" element="C" mass="12.011" name="C502"/>
    <Type class="CR" element="C" mass="12.011" name="C506"/>
    <Type class="CR" element="C" mass="12.011" name="C509"/>
    <Type class="CS" element="C" mass="12.011" name="C500"/>
    <Type class="CT" element="C" mass="12.011" name="C135"/>
    <Type class="CT" element="C" mass="12.011" name="C136"/>
    <Type class="CT" element="C" mass="12.011" name="C137"/>
    <Type class="CT" element="C" mass="12.011" name="C149"/>
    <Type class="CT" element="C" mass="12.011" name="C157"/>
    <Type class="CT" element="C" mass="12.011" name="C158"/>
    <Type class="CT" element="C" mass="12.011" name="C206"/>
    <Type class="CT" element="C" mass="12.011" name="C209"/>
    <Type class="CT" element="C" mass="12.011" name="C210"/>
    <Type class="CT" element="C" mass="12.011" name="C214"/>
    <Type class="CT" element="C" mass="12.011" name="C223"/>
    <!-- from c-alfa renaming -->
    <Type class="CT" element="C" mass="12.011" name="C224"/>
    <Type class="CT" element="C" mass="12.011" name="C224A"/>
    <Type class="CT" element="C" mass="12.011" name="C224I"/>
    <Type class="CT" element="C" mass="12.011" name="C224S"/>
    <Type class="CT" element="C" mass="12.011" name="C224K"/>
    <Type class="CT" element="C" mass="12.011" name="C224Y"/>
    <!-- from c-alfa DONE -->
    <Type class="CT" element="C" mass="12.011" name="C242"/>
    <Type class="CT" element="C" mass="12.011" name="C245"/>
    <Type class="CT" element="C" mass="12.011" name="C246"/>
    <Type class="CT" element="C" mass="12.011" name="C274"/>
    <Type class="CT" element="C" mass="12.011" name="C283"/>
    <Type class="CT" element="C" mass="12.011" name="C284"/>
    <Type class="CT" element="C" mass="12.011" name="C285"/>
    <Type class="CT" element="C" mass="12.011" name="C292"/>
    <Type class="CT" element="C" mass="12.011" name="C293"/>
    <Type class="CT" element="C" mass="12.011" name="C295"/>
    <Type class="CT" element="C" mass="12.011" name="C296"/>
    <Type class="CT" element="C" mass="12.011" name="C307"/>
    <Type class="CT" element="C" mass="12.011" name="C308"/>
    <Type class="CT" element="C" mass="12.011" name="C505"/>
    <Type class="CV" element="C" mass="12.011" name="C507"/>
    <Type class="CW" element="C" mass="12.011" name="C508"/>
    <Type class="CW" element="C" mass="12.011" name="C514"/>
    <Type class="CX" element="C" mass="12.011" name="C510"/>
    <Type class="H" element="H" mass="1.008" name="H240"/>
    <Type class="H" element="H" mass="1.008" name="H241"/>
    <Type class="H" element="H" mass="1.008" name="H504"/>
    <Type class="H" element="H" mass="1.008" name="H513"/>
    <Type class="H3" element="H" mass="1.008" name="H290"/>
    <Type class="H3" element="H" mass="1.008" name="H301"/>
    <Type class="H3" element="H" mass="1.008" name="H304"/>
    <Type class="H3" element="H" mass="1.008" name="H310"/>
    <Type class="HA" element="H" mass="1.008" name="H146"/>
    <Type class="HC" element="H" mass="1.008" name="H140"/>
    <Type class="HO" element="H" mass="1.008" name="H155"/>
    <Type class="HO" element="H" mass="1.008" name="H168"/>
    <Type class="HO" element="H" mass="1.008" name="H270"/>
    <Type class="HS" element="H" mass="1.008" name="H204"/>
    <Type class="N" element="N" mass="14.007" name="N237"/>
    <Type class="N" element="N" mass="14.007" name="N238"/>
    <Type class="N" element="N" mass="14.007" name="N239"/>
    <Type class="N2" element="N" mass="14.007" name="N300"/>
    <Type class="N2" element="N" mass="14.007" name="N303"/>
    <Type class="N3" element="N" mass="14.007" name="N287"/>
    <Type class="N3" element="N" mass="14.007" name="N309"/>
    <Type class="NA" element="N" mass="14.007" name="N503"/>
    <Type class="NA" element="N" mass="14.007" name="N512"/>
    <Type class="NB" element="N" mass="14.007" name="N511"/>
    <Type class="O" element="O" mass="15.9994" name="O236"/>
    <Type class="O" element="O" mass="15.9994" name="O269"/>
    <Type class="O2" element="O" mass="15.9994" name="O272"/>
    <Type class="OH" element="O" mass="15.9994" name="O154"/>
    <Type class="OH" element="O" mass="15.9994" name="O167"/>
    <Type class="OH" element="O" mass="15.9994" name="O268"/>
    <Type class="S" element="S" mass="32.06" name="S202"/>
    <Type class="S" element="S" mass="32.06" name="S203"/>
    <Type class="SH" element="S" mass="32.06" name="S200"/>
    <!-- from excel script -->
    <!--DATA FOR WATER    -->
    <Type class="OT" element="O" mass="15.9994" name="tip3p-O"/>
    <Type class="HT" element="H" mass="15.9994" name="tip3p-H"/>
    <!--DATA FOR IONS    -->
    <Type class="SOD" element="Na" mass="22.989800" name="SOD"/>
    <Type class="CLA" element="Cl" mass="35.450000" name="CLA"/>
  </AtomTypes>
  <Residues>
    <Residue name="TIP3">
      <Atom charge="-0.834" name="OH2" type="tip3p-O"/>
      <Atom charge="0.417" name="H1" type="tip3p-H"/>
      <Atom charge="0.417" name="H2" type="tip3p-H"/>
      <Bond atomName1="OH2" atomName2="H1"/>
      <Bond atomName1="OH2" atomName2="H2"/>
    </Residue>
    <Residue name="SOD">
      <Atom charge="1" name="SOD" type="SOD"/>
    </Residue>
    <Residue name="CLA">
      <Atom charge="-1" name="CLA" type="CLA"/>
    </Residue>
    <Residue name="ALA">
      <Atom charge="-0.4157" name="N" type="N238"/>
      <Atom charge="0.2719" name="H" type="H241"/>
      <Atom charge="0.0337" name="CA" type="C224A"/>
      <Atom charge="0.0823" name="HA" type="H140"/>
      <Atom charge="-0.1825" name="CB" type="C135"/>
      <Atom charge="0.0603" name="HB1" type="H140"/>
      <Atom charge="0.0603" name="HB2" type="H140"/>
      <Atom charge="0.0603" name="HB3" type="H140"/>
      <Atom charge="0.5973" name="C" type="C235"/>
      <Atom charge="-0.5679" name="O" type="O236"/>
      <Bond atomName1="N" atomName2="H"/>
      <Bond atomName1="N" atomName2="CA"/>
      <Bond atomName1="CA" atomName2="HA"/>
      <Bond atomName1="CA" atomName2="CB"/>
      <Bond atomName1="CA" atomName2="C"/>
      <Bond atomName1="CB" atomName2="HB1"/>
      <Bond atomName1="CB" atomName2="HB2"/>
      <Bond atomName1="CB" atomName2="HB3"/>
      <Bond atomName1="C" atomName2="O"/>
      <ExternalBond atomName="N"/>
      <ExternalBond atomName="C"/>
    </Residue>
    <Residue name="ARG">
      <Atom charge="-0.3479" name="N" type="N238"/>
      <Atom charge="0.2747" name="H" type="H241"/>
      <Atom charge="-0.2637" name="CA" type="C224"/>
      <Atom charge="0.156" name="HA" type="H140"/>
      <Atom charge="-0.0007" name="CB" type="C136"/>
      <Atom charge="0.0327" name="HB2" type="H140"/>
      <Atom charge="0.0327" name="HB3" type="H140"/>
      <Atom charge="0.039" name="CG" type="C308"/>
      <Atom charge="0.0285" name="HG2" type="H140"/>
      <Atom charge="0.0285" name="HG3" type="H140"/>
      <Atom charge="0.0486" name="CD" type="C307"/>
      <Atom charge="0.0687" name="HD2" type="H140"/>
      <Atom charge="0.0687" name="HD3" type="H140"/>
      <Atom charge="-0.5295" name="NE" type="N303"/>
      <Atom charge="0.3456" name="HE" type="H304"/>
      <Atom charge="0.8076" name="CZ" type="C302"/>
      <Atom charge="-0.8627" name="NH1" type="N300"/>
      <Atom charge="0.4478" name="HH11" type="H301"/>
      <Atom charge="0.4478" name="HH12" type="H301"/>
      <Atom charge="-0.8627" name="NH2" type="N300"/>
      <Atom charge="0.4478" name="HH21" type="H301"/>
      <Atom charge="0.4478" name="HH22" type="H301"/>
      <Atom charge="0.7341" name="C" type="C235"/>
      <Atom charge="-0.5894" name="O" type="O236"/>
      <Bond atomName1="N" atomName2="H"/>
      <Bond atomName1="N" atomName2="CA"/>
      <Bond atomName1="CA" atomName2="HA"/>
      <Bond atomName1="CA" atomName2="CB"/>
      <Bond atomName1="CA" atomName2="C"/>
      <Bond atomName1="CB" atomName2="HB2"/>
      <Bond atomName1="CB" atomName2="HB3"/>
      <Bond atomName1="CB" atomName2="CG"/>
      <Bond atomName1="CG" atomName2="HG2"/>
      <Bond atomName1="CG" atomName2="HG3"/>
      <Bond atomName1="CG" atomName2="CD"/>
      <Bond atomName1="CD" atomName2="HD2"/>
      <Bond atomName1="CD" atomName2="HD3"/>
      <Bond atomName1="CD" atomName2="NE"/>
      <Bond atomName1="NE" atomName2="HE"/>
      <Bond atomName1="NE" atomName2="CZ"/>
      <Bond atomName1="CZ" atomName2="NH1"/>
      <Bond atomName1="CZ" atomName2="NH2"/>
      <Bond atomName1="NH1" atomName2="HH11"/>
      <Bond atomName1="NH1" atomName2="HH12"/>
      <Bond atomName1="NH2" atomName2="HH21"/>
      <Bond atomName1="NH2" atomName2="HH22"/>
      <Bond atomName1="C" atomName2="O"/>
      <ExternalBond atomName="N"/>
      <ExternalBond atomName="C"/>
    </Residue>
    <Residue name="ASH">
      <Atom charge="-0.4157" name="N" type="N238"/>
      <Atom charge="0.2719" name="H" type="H241"/>
      <Atom charge="0.0341" name="CA" type="C224"/>
      <Atom charge="0.0864" name="HA" type="H140"/>
      <Atom charge="-0.0316" name="CB" type="C136"/>
      <Atom charge="0.0488" name="HB2" type="H140"/>
      <Atom charge="0.0488" name="HB3" type="H140"/>
      <Atom charge="0.6462" name="CG" type="C267"/>
      <Atom charge="-0.5554" name="OD1" type="O269"/>
      <Atom charge="-0.6376" name="OD2" type="O268"/>
      <Atom charge="0.4747" name="HD2" type="H270"/>
      <Atom charge="0.5973" name="C" type="C235"/>
      <Atom charge="-0.5679" name="O" type="O236"/>
      <Bond atomName1="N" atomName2="H"/>
      <Bond atomName1="N" atomName2="CA"/>
      <Bond atomName1="CA" atomName2="HA"/>
      <Bond atomName1="CA" atomName2="CB"/>
      <Bond atomName1="CA" atomName2="C"/>
      <Bond atomName1="CB" atomName2="HB2"/>
      <Bond atomName1="CB" atomName2="HB3"/>
      <Bond atomName1="CB" atomName2="CG"/>
      <Bond atomName1="CG" atomName2="OD1"/>
      <Bond atomName1="CG" atomName2="OD2"/>
      <Bond atomName1="OD2" atomName2="HD2"/>
      <Bond atomName1="C" atomName2="O"/>
      <ExternalBond atomName="N"/>
      <ExternalBond atomName="C"/>
    </Residue>
    <Residue name="ASN">
      <Atom charge="-0.4157" name="N" type="N238"/>
      <Atom charge="0.2719" name="H" type="H241"/>
      <Atom charge="0.0143" name="CA" type="C224"/>
      <Atom charge="0.1048" name="HA" type="H140"/>
      <Atom charge="-0.2041" name="CB" type="C136"/>
      <Atom charge="0.0797" name="HB2" type="H140"/>
      <Atom charge="0.0797" name="HB3" type="H140"/>
      <Atom charge="0.713" name="CG" type="C235"/>
      <Atom charge="-0.5931" name="OD1" type="O236"/>
      <Atom charge="-0.9191" name="ND2" type="N237"/>
      <Atom charge="0.4196" name="HD21" type="H240"/>
      <Atom charge="0.4196" name="HD22" type="H240"/>
      <Atom charge="0.5973" name="C" type="C235"/>
      <Atom charge="-0.5679" name="O" type="O236"/>
      <Bond atomName1="N" atomName2="H"/>
      <Bond atomName1="N" atomName2="CA"/>
      <Bond atomName1="CA" atomName2="HA"/>
      <Bond atomName1="CA" atomName2="CB"/>
      <Bond atomName1="CA" atomName2="C"/>
      <Bond atomName1="CB" atomName2="HB2"/>
      <Bond atomName1="CB" atomName2="HB3"/>
      <Bond atomName1="CB" atomName2="CG"/>
      <Bond atomName1="CG" atomName2="OD1"/>
      <Bond atomName1="CG" atomName2="ND2"/>
      <Bond atomName1="ND2" atomName2="HD21"/>
      <Bond atomName1="ND2" atomName2="HD22"/>
      <Bond atomName1="C" atomName2="O"/>
      <ExternalBond atomName="N"/>
      <ExternalBond atomName="C"/>
    </Residue>
    <Residue name="ASP">
      <Atom charge="-0.5163" name="N" type="N238"/>
      <Atom charge="0.2936" name="H" type="H241"/>
      <Atom charge="0.0381" name="CA" type="C224"/>
      <Atom charge="0.088" name="HA" type="H140"/>
      <Atom charge="-0.0303" name="CB" type="C274"/>
      <Atom charge="-0.0122" name="HB2" type="H140"/>
      <Atom charge="-0.0122" name="HB3" type="H140"/>
      <Atom charge="0.7994" name="CG" type="C271"/>
      <Atom charge="-0.8014" name="OD1" type="O272"/>
      <Atom charge="-0.8014" name="OD2" type="O272"/>
      <Atom charge="0.5366" name="C" type="C235"/>
      <Atom charge="-0.5819" name="O" type="O236"/>
      <Bond atomName1="N" atomName2="H"/>
      <Bond atomName1="N" atomName2="CA"/>
      <Bond atomName1="CA" atomName2="HA"/>
      <Bond atomName1="CA" atomName2="CB"/>
      <Bond atomName1="CA" atomName2="C"/>
      <Bond atomName1="CB" atomName2="HB2"/>
      <Bond atomName1="CB" atomName2="HB3"/>
      <Bond atomName1="CB" atomName2="CG"/>
      <Bond atomName1="CG" atomName2="OD1"/>
      <Bond atomName1="CG" atomName2="OD2"/>
      <Bond atomName1="C" atomName2="O"/>
      <ExternalBond atomName="N"/>
      <ExternalBond atomName="C"/>
    </Residue>
    <Residue name="CYS">
      <Atom charge="-0.4157" name="N" type="N238"/>
      <Atom charge="0.2719" name="H" type="H241"/>
      <Atom charge="0.0213" name="CA" type="C224"/>
      <Atom charge="0.1124" name="HA" type="H140"/>
      <Atom charge="-0.1231" name="CB" type="C206"/>
      <Atom charge="0.1112" name="HB2" type="H140"/>
      <Atom charge="0.1112" name="HB3" type="H140"/>
      <Atom charge="-0.3119" name="SG" type="S200"/>
      <Atom charge="0.1933" name="HG" type="H204"/>
      <Atom charge="0.5973" name="C" type="C235"/>
      <Atom charge="-0.5679" name="O" type="O236"/>
      <Bond atomName1="N" atomName2="H"/>
      <Bond atomName1="N" atomName2="CA"/>
      <Bond atomName1="CA" atomName2="HA"/>
      <Bond atomName1="CA" atomName2="CB"/>
      <Bond atomName1="CA" atomName2="C"/>
      <Bond atomName1="CB" atomName2="HB2"/>
      <Bond atomName1="CB" atomName2="HB3"/>
      <Bond atomName1="CB" atomName2="SG"/>
      <Bond atomName1="SG" atomName2="HG"/>
      <Bond atomName1="C" atomName2="O"/>
      <ExternalBond atomName="N"/>
      <ExternalBond atomName="C"/>
    </Residue>
    <Residue name="CYX">
      <Atom charge="-0.4157" name="N" type="N238"/>
      <Atom charge="0.2719" name="H" type="H241"/>
      <Atom charge="0.0429" name="CA" type="C224"/>
      <Atom charge="0.0766" name="HA" type="H140"/>
      <Atom charge="-0.079" name="CB" type="C214"/>
      <Atom charge="0.091" name="HB2" type="H140"/>
      <Atom charge="0.091" name="HB3" type="H140"/>
      <Atom charge="-0.1081" name="SG" type="S203"/>
      <Atom charge="0.5973" name="C" type="C235"/>
      <Atom charge="-0.5679" name="O" type="O236"/>
      <Bond atomName1="N" atomName2="H"/>
      <Bond atomName1="N" atomName2="CA"/>
      <Bond atomName1="CA" atomName2="HA"/>
      <Bond atomName1="CA" atomName2="CB"/>
      <Bond atomName1="CA" atomName2="C"/>
      <Bond atomName1="CB" atomName2="HB2"/>
      <Bond atomName1="CB" atomName2="HB3"/>
      <Bond atomName1="CB" atomName2="SG"/>
      <Bond atomName1="C" atomName2="O"/>
      <ExternalBond atomName="SG"/>
      <ExternalBond atomName="N"/>
      <ExternalBond atomName="C"/>
    </Residue>
    <Residue name="GLH">
      <Atom charge="-0.4157" name="N" type="N238"/>
      <Atom charge="0.2719" name="H" type="H241"/>
      <Atom charge="0.0145" name="CA" type="C224"/>
      <Atom charge="0.0779" name="HA" type="H140"/>
      <Atom charge="-0.0071" name="CB" type="C136"/>
      <Atom charge="0.0256" name="HB2" type="H140"/>
      <Atom charge="0.0256" name="HB3" type="H140"/>
      <Atom charge="-0.0174" name="CG" type="C136"/>
      <Atom charge="0.043" name="HG2" type="H140"/>
      <Atom charge="0.043" name="HG3" type="H140"/>
      <Atom charge="0.6801" name="CD" type="C267"/>
      <Atom charge="-0.5838" name="OE1" type="O269"/>
      <Atom charge="-0.6511" name="OE2" type="O268"/>
      <Atom charge="0.4641" name="HE2" type="H270"/>
      <Atom charge="0.5973" name="C" type="C235"/>
      <Atom charge="-0.5679" name="O" type="O236"/>
      <Bond atomName1="N" atomName2="H"/>
      <Bond atomName1="N" atomName2="CA"/>
      <Bond atomName1="CA" atomName2="HA"/>
      <Bond atomName1="CA" atomName2="CB"/>
      <Bond atomName1="CA" atomName2="C"/>
      <Bond atomName1="CB" atomName2="HB2"/>
      <Bond atomName1="CB" atomName2="HB3"/>
      <Bond atomName1="CB" atomName2="CG"/>
      <Bond atomName1="CG" atomName2="HG2"/>
      <Bond atomName1="CG" atomName2="HG3"/>
      <Bond atomName1="CG" atomName2="CD"/>
      <Bond atomName1="CD" atomName2="OE1"/>
      <Bond atomName1="CD" atomName2="OE2"/>
      <Bond atomName1="OE2" atomName2="HE2"/>
      <Bond atomName1="C" atomName2="O"/>
      <ExternalBond atomName="N"/>
      <ExternalBond atomName="C"/>
    </Residue>
    <Residue name="GLN">
      <Atom charge="-0.4157" name="N" type="N238"/>
      <Atom charge="0.2719" name="H" type="H241"/>
      <Atom charge="-0.0031" name="CA" type="C224"/>
      <Atom charge="0.085" name="HA" type="H140"/>
      <Atom charge="-0.0036" name="CB" type="C136"/>
      <Atom charge="0.0171" name="HB2" type="H140"/>
      <Atom charge="0.0171" name="HB3" type="H140"/>
      <Atom charge="-0.0645" name="CG" type="C136"/>
      <Atom charge="0.0352" name="HG2" type="H140"/>
      <Atom charge="0.0352" name="HG3" type="H140"/>
      <Atom charge="0.6951" name="CD" type="C235"/>
      <Atom charge="-0.6086" name="OE1" type="O236"/>
      <Atom charge="-0.9407" name="NE2" type="N237"/>
      <Atom charge="0.4251" name="HE21" type="H240"/>
      <Atom charge="0.4251" name="HE22" type="H240"/>
      <Atom charge="0.5973" name="C" type="C235"/>
      <Atom charge="-0.5679" name="O" type="O236"/>
      <Bond atomName1="N" atomName2="H"/>
      <Bond atomName1="N" atomName2="CA"/>
      <Bond atomName1="CA" atomName2="HA"/>
      <Bond atomName1="CA" atomName2="CB"/>
      <Bond atomName1="CA" atomName2="C"/>
      <Bond atomName1="CB" atomName2="HB2"/>
      <Bond atomName1="CB" atomName2="HB3"/>
      <Bond atomName1="CB" atomName2="CG"/>
      <Bond atomName1="CG" atomName2="HG2"/>
      <Bond atomName1="CG" atomName2="HG3"/>
      <Bond atomName1="CG" atomName2="CD"/>
      <Bond atomName1="CD" atomName2="OE1"/>
      <Bond atomName1="CD" atomName2="NE2"/>
      <Bond atomName1="NE2" atomName2="HE21"/>
      <Bond atomName1="NE2" atomName2="HE22"/>
      <Bond atomName1="C" atomName2="O"/>
      <ExternalBond atomName="N"/>
      <ExternalBond atomName="C"/>
    </Residue>
    <Residue name="GLU">
      <Atom charge="-0.5163" name="N" type="N238"/>
      <Atom charge="0.2936" name="H" type="H241"/>
      <Atom charge="0.0397" name="CA" type="C224"/>
      <Atom charge="0.1105" name="HA" type="H140"/>
      <Atom charge="0.056" name="CB" type="C136"/>
      <Atom charge="-0.0173" name="HB2" type="H140"/>
      <Atom charge="-0.0173" name="HB3" type="H140"/>
      <Atom charge="0.0136" name="CG" type="C274"/>
      <Atom charge="-0.0425" name="HG2" type="H140"/>
      <Atom charge="-0.0425" name="HG3" type="H140"/>
      <Atom charge="0.8054" name="CD" type="C271"/>
      <Atom charge="-0.8188" name="OE1" type="O272"/>
      <Atom charge="-0.8188" name="OE2" type="O272"/>
      <Atom charge="0.5366" name="C" type="C235"/>
      <Atom charge="-0.5819" name="O" type="O236"/>
      <Bond atomName1="N" atomName2="H"/>
      <Bond atomName1="N" atomName2="CA"/>
      <Bond atomName1="CA" atomName2="HA"/>
      <Bond atomName1="CA" atomName2="CB"/>
      <Bond atomName1="CA" atomName2="C"/>
      <Bond atomName1="CB" atomName2="HB2"/>
      <Bond atomName1="CB" atomName2="HB3"/>
      <Bond atomName1="CB" atomName2="CG"/>
      <Bond atomName1="CG" atomName2="HG2"/>
      <Bond atomName1="CG" atomName2="HG3"/>
      <Bond atomName1="CG" atomName2="CD"/>
      <Bond atomName1="CD" atomName2="OE1"/>
      <Bond atomName1="CD" atomName2="OE2"/>
      <Bond atomName1="C" atomName2="O"/>
      <ExternalBond atomName="N"/>
      <ExternalBond atomName="C"/>
    </Residue>
    <Residue name="GLY">
      <Atom charge="-0.4157" name="N" type="N238"/>
      <Atom charge="0.2719" name="H" type="H241"/>
      <Atom charge="-0.0252" name="CA" type="C223"/>
      <Atom charge="0.0698" name="HA2" type="H140"/>
      <Atom charge="0.0698" name="HA3" type="H140"/>
      <Atom charge="0.5973" name="C" type="C235"/>
      <Atom charge="-0.5679" name="O" type="O236"/>
      <Bond atomName1="N" atomName2="H"/>
      <Bond atomName1="N" atomName2="CA"/>
      <Bond atomName1="CA" atomName2="HA2"/>
      <Bond atomName1="CA" atomName2="HA3"/>
      <Bond atomName1="CA" atomName2="C"/>
      <Bond atomName1="C" atomName2="O"/>
      <ExternalBond atomName="N"/>
      <ExternalBond atomName="C"/>
    </Residue>
    <Residue name="HID">
      <Atom charge="-0.4157" name="N" type="N238"/>
      <Atom charge="0.2719" name="H" type="H241"/>
      <Atom charge="0.0188" name="CA" type="C224"/>
      <Atom charge="0.0881" name="HA" type="H140"/>
      <Atom charge="-0.0462" name="CB" type="C505"/>
      <Atom charge="0.0402" name="HB2" type="H140"/>
      <Atom charge="0.0402" name="HB3" type="H140"/>
      <Atom charge="-0.0266" name="CG" type="C508"/>
      <Atom charge="-0.3811" name="ND1" type="N503"/>
      <Atom charge="0.3649" name="HD1" type="H504"/>
      <Atom charge="0.2057" name="CE1" type="C506"/>
      <Atom charge="0.1392" name="HE1" type="H146"/>
      <Atom charge="-0.5727" name="NE2" type="N511"/>
      <Atom charge="0.1292" name="CD2" type="C507"/>
      <Atom charge="0.1147" name="HD2" type="H146"/>
      <Atom charge="0.5973" name="C" type="C235"/>
      <Atom charge="-0.5679" name="O" type="O236"/>
      <Bond atomName1="N" atomName2="H"/>
      <Bond atomName1="N" atomName2="CA"/>
      <Bond atomName1="CA" atomName2="HA"/>
      <Bond atomName1="CA" atomName2="CB"/>
      <Bond atomName1="CA" atomName2="C"/>
      <Bond atomName1="CB" atomName2="HB2"/>
      <Bond atomName1="CB" atomName2="HB3"/>
      <Bond atomName1="CB" atomName2="CG"/>
      <Bond atomName1="CG" atomName2="ND1"/>
      <Bond atomName1="CG" atomName2="CD2"/>
      <Bond atomName1="ND1" atomName2="HD1"/>
      <Bond atomName1="ND1" atomName2="CE1"/>
      <Bond atomName1="CE1" atomName2="HE1"/>
      <Bond atomName1="CE1" atomName2="NE2"/>
      <Bond atomName1="NE2" atomName2="CD2"/>
      <Bond atomName1="CD2" atomName2="HD2"/>
      <Bond atomName1="C" atomName2="O"/>
      <ExternalBond atomName="N"/>
      <ExternalBond atomName="C"/>
    </Residue>
    <Residue name="HIE">
      <Atom charge="-0.4157" name="N" type="N238"/>
      <Atom charge="0.2719" name="H" type="H241"/>
      <Atom charge="-0.0581" name="CA" type="C224"/>
      <Atom charge="0.136" name="HA" type="H140"/>
      <Atom charge="-0.0074" name="CB" type="C505"/>
      <Atom charge="0.0367" name="HB2" type="H140"/>
      <Atom charge="0.0367" name="HB3" type="H140"/>
      <Atom charge="0.1868" name="CG" type="C507"/>
      <Atom charge="-0.5432" name="ND1" type="N511"/>
      <Atom charge="0.1635" name="CE1" type="C506"/>
      <Atom charge="0.1435" name="HE1" type="H146"/>
      <Atom charge="-0.2795" name="NE2" type="N503"/>
      <Atom charge="0.3339" name="HE2" type="H504"/>
      <Atom charge="-0.2207" name="CD2" type="C508"/>
      <Atom charge="0.1862" name="HD2" type="H146"/>
      <Atom charge="0.5973" name="C" type="C235"/>
      <Atom charge="-0.5679" name="O" type="O236"/>
      <Bond atomName1="N" atomName2="H"/>
      <Bond atomName1="N" atomName2="CA"/>
      <Bond atomName1="CA" atomName2="HA"/>
      <Bond atomName1="CA" atomName2="CB"/>
      <Bond atomName1="CA" atomName2="C"/>
      <Bond atomName1="CB" atomName2="HB2"/>
      <Bond atomName1="CB" atomName2="HB3"/>
      <Bond atomName1="CB" atomName2="CG"/>
      <Bond atomName1="CG" atomName2="ND1"/>
      <Bond atomName1="CG" atomName2="CD2"/>
      <Bond atomName1="ND1" atomName2="CE1"/>
      <Bond atomName1="CE1" atomName2="HE1"/>
      <Bond atomName1="CE1" atomName2="NE2"/>
      <Bond atomName1="NE2" atomName2="HE2"/>
      <Bond atomName1="NE2" atomName2="CD2"/>
      <Bond atomName1="CD2" atomName2="HD2"/>
      <Bond atomName1="C" atomName2="O"/>
      <ExternalBond atomName="N"/>
      <ExternalBond atomName="C"/>
    </Residue>
    <Residue name="HIP">
      <Atom charge="-0.3479" name="N" type="N238"/>
      <Atom charge="0.2747" name="H" type="H241"/>
      <Atom charge="-0.1354" name="CA" type="C224"/>
      <Atom charge="0.1212" name="HA" type="H140"/>
      <Atom charge="-0.0414" name="CB" type="C505"/>
      <Atom charge="0.081" name="HB2" type="H140"/>
      <Atom charge="0.081" name="HB3" type="H140"/>
      <Atom charge="-0.0012" name="CG" type="C510"/>
      <Atom charge="-0.1513" name="ND1" type="N512"/>
      <Atom charge="0.3866" name="HD1" type="H513"/>
      <Atom charge="-0.017" name="CE1" type="C509"/>
      <Atom charge="0.2681" name="HE1" type="H146"/>
      <Atom charge="-0.1718" name="NE2" type="N512"/>
      <Atom charge="0.3911" name="HE2" type="H513"/>
      <Atom charge="-0.1141" name="CD2" type="C510"/>
      <Atom charge="0.2317" name="HD2" type="H146"/>
      <Atom charge="0.7341" name="C" type="C235"/>
      <Atom charge="-0.5894" name="O" type="O236"/>
      <Bond atomName1="N" atomName2="H"/>
      <Bond atomName1="N" atomName2="CA"/>
      <Bond atomName1="CA" atomName2="HA"/>
      <Bond atomName1="CA" atomName2="CB"/>
      <Bond atomName1="CA" atomName2="C"/>
      <Bond atomName1="CB" atomName2="HB2"/>
      <Bond atomName1="CB" atomName2="HB3"/>
      <Bond atomName1="CB" atomName2="CG"/>
      <Bond atomName1="CG" atomName2="ND1"/>
      <Bond atomName1="CG" atomName2="CD2"/>
      <Bond atomName1="ND1" atomName2="HD1"/>
      <Bond atomName1="ND1" atomName2="CE1"/>
      <Bond atomName1="CE1" atomName2="HE1"/>
      <Bond atomName1="CE1" atomName2="NE2"/>
      <Bond atomName1="NE2" atomName2="HE2"/>
      <Bond atomName1="NE2" atomName2="CD2"/>
      <Bond atomName1="CD2" atomName2="HD2"/>
      <Bond atomName1="C" atomName2="O"/>
      <ExternalBond atomName="N"/>
      <ExternalBond atomName="C"/>
    </Residue>
    <Residue name="ILE">
      <Atom charge="-0.4157" name="N" type="N238"/>
      <Atom charge="0.2719" name="H" type="H241"/>
      <Atom charge="-0.0597" name="CA" type="C224I"/>
      <Atom charge="0.0869" name="HA" type="H140"/>
      <Atom charge="0.1303" name="CB" type="C137"/>
      <Atom charge="0.0187" name="HB" type="H140"/>
      <Atom charge="-0.3204" name="CG2" type="C135"/>
      <Atom charge="0.0882" name="HG21" type="H140"/>
      <Atom charge="0.0882" name="HG22" type="H140"/>
      <Atom charge="0.0882" name="HG23" type="H140"/>
      <Atom charge="-0.043" name="CG1" type="C136"/>
      <Atom charge="0.0236" name="HG12" type="H140"/>
      <Atom charge="0.0236" name="HG13" type="H140"/>
      <Atom charge="-0.066" name="CD1" type="C135"/>
      <Atom charge="0.0186" name="HD11" type="H140"/>
      <Atom charge="0.0186" name="HD12" type="H140"/>
      <Atom charge="0.0186" name="HD13" type="H140"/>
      <Atom charge="0.5973" name="C" type="C235"/>
      <Atom charge="-0.5679" name="O" type="O236"/>
      <Bond atomName1="N" atomName2="H"/>
      <Bond atomName1="N" atomName2="CA"/>
      <Bond atomName1="CA" atomName2="HA"/>
      <Bond atomName1="CA" atomName2="CB"/>
      <Bond atomName1="CA" atomName2="C"/>
      <Bond atomName1="CB" atomName2="HB"/>
      <Bond atomName1="CB" atomName2="CG2"/>
      <Bond atomName1="CB" atomName2="CG1"/>
      <Bond atomName1="CG2" atomName2="HG21"/>
      <Bond atomName1="CG2" atomName2="HG22"/>
      <Bond atomName1="CG2" atomName2="HG23"/>
      <Bond atomName1="CG1" atomName2="HG12"/>
      <Bond atomName1="CG1" atomName2="HG13"/>
      <Bond atomName1="CG1" atomName2="CD1"/>
      <Bond atomName1="CD1" atomName2="HD11"/>
      <Bond atomName1="CD1" atomName2="HD12"/>
      <Bond atomName1="CD1" atomName2="HD13"/>
      <Bond atomName1="C" atomName2="O"/>
      <ExternalBond atomName="N"/>
      <ExternalBond atomName="C"/>
    </Residue>
    <Residue name="LEU">
      <Atom charge="-0.6104" name="N" type="N238"/>
      <Atom charge="0.3534" name="H" type="H241"/>
      <Atom charge="0.1797" name="CA" type="C224"/>
      <Atom charge="0.0778" name="HA" type="H140"/>
      <Atom charge="-0.2488" name="CB" type="C136"/>
      <Atom charge="0.0848" name="HB2" type="H140"/>
      <Atom charge="0.0637" name="HB3" type="H140"/>
      <Atom charge="0.3273" name="CG" type="C137"/>
      <Atom charge="-0.0217" name="HG" type="H140"/>
      <Atom charge="-0.3691" name="CD1" type="C135"/>
      <Atom charge="0.0868" name="HD11" type="H140"/>
      <Atom charge="0.0868" name="HD12" type="H140"/>
      <Atom charge="0.0868" name="HD13" type="H140"/>
      <Atom charge="-0.3739" name="CD2" type="C135"/>
      <Atom charge="0.0939" name="HD21" type="H140"/>
      <Atom charge="0.0939" name="HD22" type="H140"/>
      <Atom charge="0.0939" name="HD23" type="H140"/>
      <Atom charge="0.4661" name="C" type="C235"/>
      <Atom charge="-0.6200" name="O" type="O236"/>
      <Bond atomName1="N" atomName2="H"/>
      <Bond atomName1="N" atomName2="CA"/>
      <Bond atomName1="CA" atomName2="HA"/>
      <Bond atomName1="CA" atomName2="CB"/>
      <Bond atomName1="CA" atomName2="C"/>
      <Bond atomName1="CB" atomName2="HB2"/>
      <Bond atomName1="CB" atomName2="HB3"/>
      <Bond atomName1="CB" atomName2="CG"/>
      <Bond atomName1="CG" atomName2="HG"/>
      <Bond atomName1="CG" atomName2="CD1"/>
      <Bond atomName1="CG" atomName2="CD2"/>
      <Bond atomName1="CD1" atomName2="HD11"/>
      <Bond atomName1="CD1" atomName2="HD12"/>
      <Bond atomName1="CD1" atomName2="HD13"/>
      <Bond atomName1="CD2" atomName2="HD21"/>
      <Bond atomName1="CD2" atomName2="HD22"/>
      <Bond atomName1="CD2" atomName2="HD23"/>
      <Bond atomName1="C" atomName2="O"/>
      <ExternalBond atomName="N"/>
      <ExternalBond atomName="C"/>
    </Residue>
    <Residue name="LYS">
      <Atom charge="-0.3479" name="N" type="N238"/>
      <Atom charge="0.2747" name="H" type="H241"/>
      <Atom charge="-0.24" name="CA" type="C224K"/>
      <Atom charge="0.1426" name="HA" type="H140"/>
      <Atom charge="-0.0094" name="CB" type="C136"/>
      <Atom charge="0.0362" name="HB2" type="H140"/>
      <Atom charge="0.0362" name="HB3" type="H140"/>
      <Atom charge="0.0187" name="CG" type="C136"/>
      <Atom charge="0.0103" name="HG2" type="H140"/>
      <Atom charge="0.0103" name="HG3" type="H140"/>
      <Atom charge="-0.0479" name="CD" type="C136"/>
      <Atom charge="0.0621" name="HD2" type="H140"/>
      <Atom charge="0.0621" name="HD3" type="H140"/>
      <Atom charge="-0.0143" name="CE" type="C292"/>
      <Atom charge="0.1135" name="HE2" type="H140"/>
      <Atom charge="0.1135" name="HE3" type="H140"/>
      <Atom charge="-0.3854" name="NZ" type="N287"/>
      <Atom charge="0.34" name="HZ1" type="H290"/>
      <Atom charge="0.34" name="HZ2" type="H290"/>
      <Atom charge="0.34" name="HZ3" type="H290"/>
      <Atom charge="0.7341" name="C" type="C235"/>
      <Atom charge="-0.5894" name="O" type="O236"/>
      <Bond atomName1="N" atomName2="H"/>
      <Bond atomName1="N" atomName2="CA"/>
      <Bond atomName1="CA" atomName2="HA"/>
      <Bond atomName1="CA" atomName2="CB"/>
      <Bond atomName1="CA" atomName2="C"/>
      <Bond atomName1="CB" atomName2="HB2"/>
      <Bond atomName1="CB" atomName2="HB3"/>
      <Bond atomName1="CB" atomName2="CG"/>
      <Bond atomName1="CG" atomName2="HG2"/>
      <Bond atomName1="CG" atomName2="HG3"/>
      <Bond atomName1="CG" atomName2="CD"/>
      <Bond atomName1="CD" atomName2="HD2"/>
      <Bond atomName1="CD" atomName2="HD3"/>
      <Bond atomName1="CD" atomName2="CE"/>
      <Bond atomName1="CE" atomName2="HE2"/>
      <Bond atomName1="CE" atomName2="HE3"/>
      <Bond atomName1="CE" atomName2="NZ"/>
      <Bond atomName1="NZ" atomName2="HZ1"/>
      <Bond atomName1="NZ" atomName2="HZ2"/>
      <Bond atomName1="NZ" atomName2="HZ3"/>
      <Bond atomName1="C" atomName2="O"/>
      <ExternalBond atomName="N"/>
      <ExternalBond atomName="C"/>
    </Residue>
    <Residue name="MET">
      <Atom charge="-0.4157" name="N" type="N238"/>
      <Atom charge="0.2719" name="H" type="H241"/>
      <Atom charge="-0.0237" name="CA" type="C224"/>
      <Atom charge="0.088" name="HA" type="H140"/>
      <Atom charge="0.0342" name="CB" type="C136"/>
      <Atom charge="0.0241" name="HB2" type="H140"/>
      <Atom charge="0.0241" name="HB3" type="H140"/>
      <Atom charge="0.0018" name="CG" type="C210"/>
      <Atom charge="0.044" name="HG2" type="H140"/>
      <Atom charge="0.044" name="HG3" type="H140"/>
      <Atom charge="-0.2737" name="SD" type="S202"/>
      <Atom charge="-0.0536" name="CE" type="C209"/>
      <Atom charge="0.0684" name="HE1" type="H140"/>
      <Atom charge="0.0684" name="HE2" type="H140"/>
      <Atom charge="0.0684" name="HE3" type="H140"/>
      <Atom charge="0.5973" name="C" type="C235"/>
      <Atom charge="-0.5679" name="O" type="O236"/>
      <Bond atomName1="N" atomName2="H"/>
      <Bond atomName1="N" atomName2="CA"/>
      <Bond atomName1="CA" atomName2="HA"/>
      <Bond atomName1="CA" atomName2="CB"/>
      <Bond atomName1="CA" atomName2="C"/>
      <Bond atomName1="CB" atomName2="HB2"/>
      <Bond atomName1="CB" atomName2="HB3"/>
      <Bond atomName1="CB" atomName2="CG"/>
      <Bond atomName1="CG" atomName2="HG2"/>
      <Bond atomName1="CG" atomName2="HG3"/>
      <Bond atomName1="CG" atomName2="SD"/>
      <Bond atomName1="SD" atomName2="CE"/>
      <Bond atomName1="CE" atomName2="HE1"/>
      <Bond atomName1="CE" atomName2="HE2"/>
      <Bond atomName1="CE" atomName2="HE3"/>
      <Bond atomName1="C" atomName2="O"/>
      <ExternalBond atomName="N"/>
      <ExternalBond atomName="C"/>
    </Residue>
    <Residue name="PHE">
      <Atom charge="-0.4157" name="N" type="N238"/>
      <Atom charge="0.2719" name="H" type="H241"/>
      <Atom charge="-0.0024" name="CA" type="C224"/>
      <Atom charge="0.0978" name="HA" type="H140"/>
      <Atom charge="-0.0343" name="CB" type="C149"/>
      <Atom charge="0.0295" name="HB2" type="H140"/>
      <Atom charge="0.0295" name="HB3" type="H140"/>
      <Atom charge="0.0118" name="CG" type="C145"/>
      <Atom charge="-0.1256" name="CD1" type="C145"/>
      <Atom charge="0.133" name="HD1" type="H146"/>
      <Atom charge="-0.1704" name="CE1" type="C145"/>
      <Atom charge="0.143" name="HE1" type="H146"/>
      <Atom charge="-0.1072" name="CZ" type="C145"/>
      <Atom charge="0.1297" name="HZ" type="H146"/>
      <Atom charge="-0.1704" name="CE2" type="C145"/>
      <Atom charge="0.143" name="HE2" type="H146"/>
      <Atom charge="-0.1256" name="CD2" type="C145"/>
      <Atom charge="0.133" name="HD2" type="H146"/>
      <Atom charge="0.5973" name="C" type="C235"/>
      <Atom charge="-0.5679" name="O" type="O236"/>
      <Bond atomName1="N" atomName2="H"/>
      <Bond atomName1="N" atomName2="CA"/>
      <Bond atomName1="CA" atomName2="HA"/>
      <Bond atomName1="CA" atomName2="CB"/>
      <Bond atomName1="CA" atomName2="C"/>
      <Bond atomName1="CB" atomName2="HB2"/>
      <Bond atomName1="CB" atomName2="HB3"/>
      <Bond atomName1="CB" atomName2="CG"/>
      <Bond atomName1="CG" atomName2="CD1"/>
      <Bond atomName1="CG" atomName2="CD2"/>
      <Bond atomName1="CD1" atomName2="HD1"/>
      <Bond atomName1="CD1" atomName2="CE1"/>
      <Bond atomName1="CE1" atomName2="HE1"/>
      <Bond atomName1="CE1" atomName2="CZ"/>
      <Bond atomName1="CZ" atomName2="HZ"/>
      <Bond atomName1="CZ" atomName2="CE2"/>
      <Bond atomName1="CE2" atomName2="HE2"/>
      <Bond atomName1="CE2" atomName2="CD2"/>
      <Bond atomName1="CD2" atomName2="HD2"/>
      <Bond atomName1="C" atomName2="O"/>
      <ExternalBond atomName="N"/>
      <ExternalBond atomName="C"/>
    </Residue>
    <Residue name="PRO">
      <Atom charge="-0.2548" name="N" type="N239"/>
      <Atom charge="0.0192" name="CD" type="C245"/>
      <Atom charge="0.0391" name="HD2" type="H140"/>
      <Atom charge="0.0391" name="HD3" type="H140"/>
      <Atom charge="0.0189" name="CG" type="C136"/>
      <Atom charge="0.0213" name="HG2" type="H140"/>
      <Atom charge="0.0213" name="HG3" type="H140"/>
      <Atom charge="-0.007" name="CB" type="C136"/>
      <Atom charge="0.0253" name="HB2" type="H140"/>
      <Atom charge="0.0253" name="HB3" type="H140"/>
      <Atom charge="-0.0266" name="CA" type="C246"/>
      <Atom charge="0.0641" name="HA" type="H140"/>
      <Atom charge="0.5896" name="C" type="C235"/>
      <Atom charge="-0.5748" name="O" type="O236"/>
      <Bond atomName1="N" atomName2="CD"/>
      <Bond atomName1="N" atomName2="CA"/>
      <Bond atomName1="CD" atomName2="HD2"/>
      <Bond atomName1="CD" atomName2="HD3"/>
      <Bond atomName1="CD" atomName2="CG"/>
      <Bond atomName1="CG" atomName2="HG2"/>
      <Bond atomName1="CG" atomName2="HG3"/>
      <Bond atomName1="CG" atomName2="CB"/>
      <Bond atomName1="CB" atomName2="HB2"/>
      <Bond atomName1="CB" atomName2="HB3"/>
      <Bond atomName1="CB" atomName2="CA"/>
      <Bond atomName1="CA" atomName2="HA"/>
      <Bond atomName1="CA" atomName2="C"/>
      <Bond atomName1="C" atomName2="O"/>
      <ExternalBond atomName="N"/>
      <ExternalBond atomName="C"/>
    </Residue>
    <Residue name="SER">
      <Atom charge="-0.4157" name="N" type="N238"/>
      <Atom charge="0.2719" name="H" type="H241"/>
      <Atom charge="-0.0249" name="CA" type="C224S"/>
      <Atom charge="0.0843" name="HA" type="H140"/>
      <Atom charge="0.2117" name="CB" type="C157"/>
      <Atom charge="0.0352" name="HB2" type="H140"/>
      <Atom charge="0.0352" name="HB3" type="H140"/>
      <Atom charge="-0.6546" name="OG" type="O154"/>
      <Atom charge="0.4275" name="HG" type="H155"/>
      <Atom charge="0.5973" name="C" type="C235"/>
      <Atom charge="-0.5679" name="O" type="O236"/>
      <Bond atomName1="N" atomName2="H"/>
      <Bond atomName1="N" atomName2="CA"/>
      <Bond atomName1="CA" atomName2="HA"/>
      <Bond atomName1="CA" atomName2="CB"/>
      <Bond atomName1="CA" atomName2="C"/>
      <Bond atomName1="CB" atomName2="HB2"/>
      <Bond atomName1="CB" atomName2="HB3"/>
      <Bond atomName1="CB" atomName2="OG"/>
      <Bond atomName1="OG" atomName2="HG"/>
      <Bond atomName1="C" atomName2="O"/>
      <ExternalBond atomName="N"/>
      <ExternalBond atomName="C"/>
    </Residue>
    <Residue name="THR">
      <Atom charge="-0.4157" name="N" type="N238"/>
      <Atom charge="0.2719" name="H" type="H241"/>
      <Atom charge="-0.0389" name="CA" type="C224S"/>
      <Atom charge="0.1007" name="HA" type="H140"/>
      <Atom charge="0.3654" name="CB" type="C158"/>
      <Atom charge="0.0043" name="HB" type="H140"/>
      <Atom charge="-0.2438" name="CG2" type="C135"/>
      <Atom charge="0.0642" name="HG21" type="H140"/>
      <Atom charge="0.0642" name="HG22" type="H140"/>
      <Atom charge="0.0642" name="HG23" type="H140"/>
      <Atom charge="-0.6761" name="OG1" type="O154"/>
      <Atom charge="0.4102" name="HG1" type="H155"/>
      <Atom charge="0.5973" name="C" type="C235"/>
      <Atom charge="-0.5679" name="O" type="O236"/>
      <Bond atomName1="N" atomName2="H"/>
      <Bond atomName1="N" atomName2="CA"/>
      <Bond atomName1="CA" atomName2="HA"/>
      <Bond atomName1="CA" atomName2="CB"/>
      <Bond atomName1="CA" atomName2="C"/>
      <Bond atomName1="CB" atomName2="HB"/>
      <Bond atomName1="CB" atomName2="CG2"/>
      <Bond atomName1="CB" atomName2="OG1"/>
      <Bond atomName1="CG2" atomName2="HG21"/>
      <Bond atomName1="CG2" atomName2="HG22"/>
      <Bond atomName1="CG2" atomName2="HG23"/>
      <Bond atomName1="OG1" atomName2="HG1"/>
      <Bond atomName1="C" atomName2="O"/>
      <ExternalBond atomName="N"/>
      <ExternalBond atomName="C"/>
    </Residue>
    <Residue name="TRP">
      <Atom charge="-0.4157" name="N" type="N238"/>
      <Atom charge="0.2719" name="H" type="H241"/>
      <Atom charge="-0.0275" name="CA" type="C224"/>
      <Atom charge="0.1123" name="HA" type="H140"/>
      <Atom charge="-0.005" name="CB" type="C136"/>
      <Atom charge="0.0339" name="HB2" type="H140"/>
      <Atom charge="0.0339" name="HB3" type="H140"/>
      <Atom charge="-0.1415" name="CG" type="C500"/>
      <Atom charge="-0.1638" name="CD1" type="C514"/>
      <Atom charge="0.2062" name="HD1" type="H146"/>
      <Atom charge="-0.3418" name="NE1" type="N503"/>
      <Atom charge="0.3412" name="HE1" type="H504"/>
      <Atom charge="0.138" name="CE2" type="C502"/>
      <Atom charge="-0.2601" name="CZ2" type="C145"/>
      <Atom charge="0.1572" name="HZ2" type="H146"/>
      <Atom charge="-0.1134" name="CH2" type="C145"/>
      <Atom charge="0.1417" name="HH2" type="H146"/>
      <Atom charge="-0.1972" name="CZ3" type="C145"/>
      <Atom charge="0.1447" name="HZ3" type="H146"/>
      <Atom charge="-0.2387" name="CE3" type="C145"/>
      <Atom charge="0.17" name="HE3" type="H146"/>
      <Atom charge="0.1243" name="CD2" type="C501"/>
      <Atom charge="0.5973" name="C" type="C235"/>
      <Atom charge="-0.5679" name="O" type="O236"/>
      <Bond atomName1="N" atomName2="H"/>
      <Bond atomName1="N" atomName2="CA"/>
      <Bond atomName1="CA" atomName2="HA"/>
      <Bond atomName1="CA" atomName2="CB"/>
      <Bond atomName1="CA" atomName2="C"/>
      <Bond atomName1="CB" atomName2="HB2"/>
      <Bond atomName1="CB" atomName2="HB3"/>
      <Bond atomName1="CB" atomName2="CG"/>
      <Bond atomName1="CG" atomName2="CD1"/>
      <Bond atomName1="CG" atomName2="CD2"/>
      <Bond atomName1="CD1" atomName2="HD1"/>
      <Bond atomName1="CD1" atomName2="NE1"/>
      <Bond atomName1="NE1" atomName2="HE1"/>
      <Bond atomName1="NE1" atomName2="CE2"/>
      <Bond atomName1="CE2" atomName2="CZ2"/>
      <Bond atomName1="CE2" atomName2="CD2"/>
      <Bond atomName1="CZ2" atomName2="HZ2"/>
      <Bond atomName1="CZ2" atomName2="CH2"/>
      <Bond atomName1="CH2" atomName2="HH2"/>
      <Bond atomName1="CH2" atomName2="CZ3"/>
      <Bond atomName1="CZ3" atomName2="HZ3"/>
      <Bond atomName1="CZ3" atomName2="CE3"/>
      <Bond atomName1="CE3" atomName2="HE3"/>
      <Bond atomName1="CE3" atomName2="CD2"/>
      <Bond atomName1="C" atomName2="O"/>
      <ExternalBond atomName="N"/>
      <ExternalBond atomName="C"/>
    </Residue>
    <Residue name="TYR">
      <Atom charge="-0.4157" name="N" type="N238"/>
      <Atom charge="0.2719" name="H" type="H241"/>
      <Atom charge="-0.0014" name="CA" type="C224Y"/>
      <Atom charge="0.0876" name="HA" type="H140"/>
      <Atom charge="-0.0152" name="CB" type="C149"/>
      <Atom charge="0.0295" name="HB2" type="H140"/>
      <Atom charge="0.0295" name="HB3" type="H140"/>
      <Atom charge="-0.0011" name="CG" type="C145"/>
      <Atom charge="-0.1906" name="CD1" type="C145"/>
      <Atom charge="0.1699" name="HD1" type="H146"/>
      <Atom charge="-0.2341" name="CE1" type="C145"/>
      <Atom charge="0.1656" name="HE1" type="H146"/>
      <Atom charge="0.3226" name="CZ" type="C166"/>
      <Atom charge="-0.5579" name="OH" type="O167"/>
      <Atom charge="0.3992" name="HH" type="H168"/>
      <Atom charge="-0.2341" name="CE2" type="C145"/>
      <Atom charge="0.1656" name="HE2" type="H146"/>
      <Atom charge="-0.1906" name="CD2" type="C145"/>
      <Atom charge="0.1699" name="HD2" type="H146"/>
      <Atom charge="0.5973" name="C" type="C235"/>
      <Atom charge="-0.5679" name="O" type="O236"/>
      <Bond atomName1="N" atomName2="H"/>
      <Bond atomName1="N" atomName2="CA"/>
      <Bond atomName1="CA" atomName2="HA"/>
      <Bond atomName1="CA" atomName2="CB"/>
      <Bond atomName1="CA" atomName2="C"/>
      <Bond atomName1="CB" atomName2="HB2"/>
      <Bond atomName1="CB" atomName2="HB3"/>
      <Bond atomName1="CB" atomName2="CG"/>
      <Bond atomName1="CG" atomName2="CD1"/>
      <Bond atomName1="CG" atomName2="CD2"/>
      <Bond atomName1="CD1" atomName2="HD1"/>
      <Bond atomName1="CD1" atomName2="CE1"/>
      <Bond atomName1="CE1" atomName2="HE1"/>
      <Bond atomName1="CE1" atomName2="CZ"/>
      <Bond atomName1="CZ" atomName2="OH"/>
      <Bond atomName1="CZ" atomName2="CE2"/>
      <Bond atomName1="OH" atomName2="HH"/>
      <Bond atomName1="CE2" atomName2="HE2"/>
      <Bond atomName1="CE2" atomName2="CD2"/>
      <Bond atomName1="CD2" atomName2="HD2"/>
      <Bond atomName1="C" atomName2="O"/>
      <ExternalBond atomName="N"/>
      <ExternalBond atomName="C"/>
    </Residue>
    <Residue name="VAL">
      <Atom charge="-0.4157" name="N" type="N238"/>
      <Atom charge="0.2719" name="H" type="H241"/>
      <Atom charge="-0.0875" name="CA" type="C224"/>
      <Atom charge="0.0969" name="HA" type="H140"/>
      <Atom charge="0.2985" name="CB" type="C137"/>
      <Atom charge="-0.0297" name="HB" type="H140"/>
      <Atom charge="-0.3192" name="CG1" type="C135"/>
      <Atom charge="0.0791" name="HG11" type="H140"/>
      <Atom charge="0.0791" name="HG12" type="H140"/>
      <Atom charge="0.0791" name="HG13" type="H140"/>
      <Atom charge="-0.3192" name="CG2" type="C135"/>
      <Atom charge="0.0791" name="HG21" type="H140"/>
      <Atom charge="0.0791" name="HG22" type="H140"/>
      <Atom charge="0.0791" name="HG23" type="H140"/>
      <Atom charge="0.5973" name="C" type="C235"/>
      <Atom charge="-0.5679" name="O" type="O236"/>
      <Bond atomName1="N" atomName2="H"/>
      <Bond atomName1="N" atomName2="CA"/>
      <Bond atomName1="CA" atomName2="HA"/>
      <Bond atomName1="CA" atomName2="CB"/>
      <Bond atomName1="CA" atomName2="C"/>
      <Bond atomName1="CB" atomName2="HB"/>
      <Bond atomName1="CB" atomName2="CG1"/>
      <Bond atomName1="CB" atomName2="CG2"/>
      <Bond atomName1="CG1" atomName2="HG11"/>
      <Bond atomName1="CG1" atomName2="HG12"/>
      <Bond atomName1="CG1" atomName2="HG13"/>
      <Bond atomName1="CG2" atomName2="HG21"/>
      <Bond atomName1="CG2" atomName2="HG22"/>
      <Bond atomName1="CG2" atomName2="HG23"/>
      <Bond atomName1="C" atomName2="O"/>
      <ExternalBond atomName="N"/>
      <ExternalBond atomName="C"/>
    </Residue>
    <Residue name="CALA">
      <Atom charge="-0.3821" name="N" type="N238"/>
      <Atom charge="0.2681" name="H" type="H241"/>
      <Atom charge="-0.1747" name="CA" type="C283"/>
      <Atom charge="0.1067" name="HA" type="H140"/>
      <Atom charge="-0.2093" name="CB" type="C135"/>
      <Atom charge="0.0764" name="HB1" type="H140"/>
      <Atom charge="0.0764" name="HB2" type="H140"/>
      <Atom charge="0.0764" name="HB3" type="H140"/>
      <Atom charge="0.7731" name="C" type="C271"/>
      <Atom charge="-0.8055" name="O" type="O272"/>
      <Atom charge="-0.8055" name="OXT" type="O272"/>
      <Bond atomName1="N" atomName2="H"/>
      <Bond atomName1="N" atomName2="CA"/>
      <Bond atomName1="CA" atomName2="HA"/>
      <Bond atomName1="CA" atomName2="CB"/>
      <Bond atomName1="CA" atomName2="C"/>
      <Bond atomName1="CB" atomName2="HB1"/>
      <Bond atomName1="CB" atomName2="HB2"/>
      <Bond atomName1="CB" atomName2="HB3"/>
      <Bond atomName1="C" atomName2="O"/>
      <Bond atomName1="C" atomName2="OXT"/>
      <ExternalBond atomName="N"/>
    </Residue>
    <Residue name="CARG">
      <Atom charge="-0.3481" name="N" type="N238"/>
      <Atom charge="0.2764" name="H" type="H241"/>
      <Atom charge="-0.3068" name="CA" type="C283"/>
      <Atom charge="0.1447" name="HA" type="H140"/>
      <Atom charge="-0.0374" name="CB" type="C136"/>
      <Atom charge="0.0371" name="HB2" type="H140"/>
      <Atom charge="0.0371" name="HB3" type="H140"/>
      <Atom charge="0.0744" name="CG" type="C308"/>
      <Atom charge="0.0185" name="HG2" type="H140"/>
      <Atom charge="0.0185" name="HG3" type="H140"/>
      <Atom charge="0.1114" name="CD" type="C307"/>
      <Atom charge="0.0468" name="HD2" type="H140"/>
      <Atom charge="0.0468" name="HD3" type="H140"/>
      <Atom charge="-0.5564" name="NE" type="N303"/>
      <Atom charge="0.3479" name="HE" type="H304"/>
      <Atom charge="0.8368" name="CZ" type="C302"/>
      <Atom charge="-0.8737" name="NH1" type="N300"/>
      <Atom charge="0.4493" name="HH11" type="H301"/>
      <Atom charge="0.4493" name="HH12" type="H301"/>
      <Atom charge="-0.8737" name="NH2" type="N300"/>
      <Atom charge="0.4493" name="HH21" type="H301"/>
      <Atom charge="0.4493" name="HH22" type="H301"/>
      <Atom charge="0.8557" name="C" type="C271"/>
      <Atom charge="-0.8266" name="O" type="O272"/>
      <Atom charge="-0.8266" name="OXT" type="O272"/>
      <Bond atomName1="N" atomName2="H"/>
      <Bond atomName1="N" atomName2="CA"/>
      <Bond atomName1="CA" atomName2="HA"/>
      <Bond atomName1="CA" atomName2="CB"/>
      <Bond atomName1="CA" atomName2="C"/>
      <Bond atomName1="CB" atomName2="HB2"/>
      <Bond atomName1="CB" atomName2="HB3"/>
      <Bond atomName1="CB" atomName2="CG"/>
      <Bond atomName1="CG" atomName2="HG2"/>
      <Bond atomName1="CG" atomName2="HG3"/>
      <Bond atomName1="CG" atomName2="CD"/>
      <Bond atomName1="CD" atomName2="HD2"/>
      <Bond atomName1="CD" atomName2="HD3"/>
      <Bond atomName1="CD" atomName2="NE"/>
      <Bond atomName1="NE" atomName2="HE"/>
      <Bond atomName1="NE" atomName2="CZ"/>
      <Bond atomName1="CZ" atomName2="NH1"/>
      <Bond atomName1="CZ" atomName2="NH2"/>
      <Bond atomName1="NH1" atomName2="HH11"/>
      <Bond atomName1="NH1" atomName2="HH12"/>
      <Bond atomName1="NH2" atomName2="HH21"/>
      <Bond atomName1="NH2" atomName2="HH22"/>
      <Bond atomName1="C" atomName2="O"/>
      <Bond atomName1="C" atomName2="OXT"/>
      <ExternalBond atomName="N"/>
    </Residue>
    <Residue name="CASH">
      <Atom charge="-0.4157" name="N" type="N238"/>
      <Atom charge="0.2719" name="H" type="H241"/>
      <Atom charge="0.0341" name="CA" type="C283"/>
      <Atom charge="0.0864" name="HA" type="H140"/>
      <Atom charge="-0.0316" name="CB" type="C136"/>
      <Atom charge="0.0488" name="HB2" type="H140"/>
      <Atom charge="0.0488" name="HB3" type="H140"/>
      <Atom charge="0.6462" name="CG" type="C267"/>
      <Atom charge="-0.5554" name="OD1" type="O269"/>
      <Atom charge="-0.6376" name="OD2" type="O268"/>
      <Atom charge="0.4747" name="HD2" type="H270"/>
      <Atom charge="0.805" name="C" type="C271"/>
      <Atom charge="-0.8147" name="O" type="O272"/>
      <Atom charge="-0.8147" name="OXT" type="O272"/>
      <Bond atomName1="N" atomName2="H"/>
      <Bond atomName1="N" atomName2="CA"/>
      <Bond atomName1="CA" atomName2="HA"/>
      <Bond atomName1="CA" atomName2="CB"/>
      <Bond atomName1="CA" atomName2="C"/>
      <Bond atomName1="CB" atomName2="HB2"/>
      <Bond atomName1="CB" atomName2="HB3"/>
      <Bond atomName1="CB" atomName2="CG"/>
      <Bond atomName1="CG" atomName2="OD1"/>
      <Bond atomName1="CG" atomName2="OD2"/>
      <Bond atomName1="OD2" atomName2="HD2"/>
      <Bond atomName1="C" atomName2="O"/>
      <Bond atomName1="C" atomName2="OXT"/>
      <ExternalBond atomName="N"/>
    </Residue>
    <Residue name="CASN">
      <Atom charge="-0.3821" name="N" type="N238"/>
      <Atom charge="0.2681" name="H" type="H241"/>
      <Atom charge="-0.208" name="CA" type="C283"/>
      <Atom charge="0.1358" name="HA" type="H140"/>
      <Atom charge="-0.2299" name="CB" type="C136"/>
      <Atom charge="0.1023" name="HB2" type="H140"/>
      <Atom charge="0.1023" name="HB3" type="H140"/>
      <Atom charge="0.7153" name="CG" type="C235"/>
      <Atom charge="-0.601" name="OD1" type="O236"/>
      <Atom charge="-0.9084" name="ND2" type="N237"/>
      <Atom charge="0.415" name="HD21" type="H240"/>
      <Atom charge="0.415" name="HD22" type="H240"/>
      <Atom charge="0.805" name="C" type="C271"/>
      <Atom charge="-0.8147" name="O" type="O272"/>
      <Atom charge="-0.8147" name="OXT" type="O272"/>
      <Bond atomName1="N" atomName2="H"/>
      <Bond atomName1="N" atomName2="CA"/>
      <Bond atomName1="CA" atomName2="HA"/>
      <Bond atomName1="CA" atomName2="CB"/>
      <Bond atomName1="CA" atomName2="C"/>
      <Bond atomName1="CB" atomName2="HB2"/>
      <Bond atomName1="CB" atomName2="HB3"/>
      <Bond atomName1="CB" atomName2="CG"/>
      <Bond atomName1="CG" atomName2="OD1"/>
      <Bond atomName1="CG" atomName2="ND2"/>
      <Bond atomName1="ND2" atomName2="HD21"/>
      <Bond atomName1="ND2" atomName2="HD22"/>
      <Bond atomName1="C" atomName2="O"/>
      <Bond atomName1="C" atomName2="OXT"/>
      <ExternalBond atomName="N"/>
    </Residue>
    <Residue name="CASP">
      <Atom charge="-0.5192" name="N" type="N238"/>
      <Atom charge="0.3055" name="H" type="H241"/>
      <Atom charge="-0.1817" name="CA" type="C283"/>
      <Atom charge="0.1046" name="HA" type="H140"/>
      <Atom charge="-0.0677" name="CB" type="C274"/>
      <Atom charge="-0.0212" name="HB2" type="H140"/>
      <Atom charge="-0.0212" name="HB3" type="H140"/>
      <Atom charge="0.8851" name="CG" type="C271"/>
      <Atom charge="-0.8162" name="OD1" type="O272"/>
      <Atom charge="-0.8162" name="OD2" type="O272"/>
      <Atom charge="0.7256" name="C" type="C271"/>
      <Atom charge="-0.7887" name="O" type="O272"/>
      <Atom charge="-0.7887" name="OXT" type="O272"/>
      <Bond atomName1="N" atomName2="H"/>
      <Bond atomName1="N" atomName2="CA"/>
      <Bond atomName1="CA" atomName2="HA"/>
      <Bond atomName1="CA" atomName2="CB"/>
      <Bond atomName1="CA" atomName2="C"/>
      <Bond atomName1="CB" atomName2="HB2"/>
      <Bond atomName1="CB" atomName2="HB3"/>
      <Bond atomName1="CB" atomName2="CG"/>
      <Bond atomName1="CG" atomName2="OD1"/>
      <Bond atomName1="CG" atomName2="OD2"/>
      <Bond atomName1="C" atomName2="O"/>
      <Bond atomName1="C" atomName2="OXT"/>
      <ExternalBond atomName="N"/>
    </Residue>
    <Residue name="CCYS">
      <Atom charge="-0.3821" name="N" type="N238"/>
      <Atom charge="0.2681" name="H" type="H241"/>
      <Atom charge="-0.1635" name="CA" type="C283"/>
      <Atom charge="0.1396" name="HA" type="H140"/>
      <Atom charge="-0.1996" name="CB" type="C206"/>
      <Atom charge="0.1437" name="HB2" type="H140"/>
      <Atom charge="0.1437" name="HB3" type="H140"/>
      <Atom charge="-0.3102" name="SG" type="S200"/>
      <Atom charge="0.2068" name="HG" type="H204"/>
      <Atom charge="0.7497" name="C" type="C271"/>
      <Atom charge="-0.7981" name="O" type="O272"/>
      <Atom charge="-0.7981" name="OXT" type="O272"/>
      <Bond atomName1="N" atomName2="H"/>
      <Bond atomName1="N" atomName2="CA"/>
      <Bond atomName1="CA" atomName2="HA"/>
      <Bond atomName1="CA" atomName2="CB"/>
      <Bond atomName1="CA" atomName2="C"/>
      <Bond atomName1="CB" atomName2="HB2"/>
      <Bond atomName1="CB" atomName2="HB3"/>
      <Bond atomName1="CB" atomName2="SG"/>
      <Bond atomName1="SG" atomName2="HG"/>
      <Bond atomName1="C" atomName2="O"/>
      <Bond atomName1="C" atomName2="OXT"/>
      <ExternalBond atomName="N"/>
    </Residue>
    <Residue name="CCYX">
      <Atom charge="-0.3821" name="N" type="N238"/>
      <Atom charge="0.2681" name="H" type="H241"/>
      <Atom charge="-0.1318" name="CA" type="C283"/>
      <Atom charge="0.0938" name="HA" type="H140"/>
      <Atom charge="-0.1943" name="CB" type="C214"/>
      <Atom charge="0.1228" name="HB2" type="H140"/>
      <Atom charge="0.1228" name="HB3" type="H140"/>
      <Atom charge="-0.0529" name="SG" type="S203"/>
      <Atom charge="0.7618" name="C" type="C271"/>
      <Atom charge="-0.8041" name="O" type="O272"/>
      <Atom charge="-0.8041" name="OXT" type="O272"/>
      <Bond atomName1="N" atomName2="H"/>
      <Bond atomName1="N" atomName2="CA"/>
      <Bond atomName1="CA" atomName2="HA"/>
      <Bond atomName1="CA" atomName2="CB"/>
      <Bond atomName1="CA" atomName2="C"/>
      <Bond atomName1="CB" atomName2="HB2"/>
      <Bond atomName1="CB" atomName2="HB3"/>
      <Bond atomName1="CB" atomName2="SG"/>
      <Bond atomName1="C" atomName2="O"/>
      <Bond atomName1="C" atomName2="OXT"/>
      <ExternalBond atomName="SG"/>
      <ExternalBond atomName="N"/>
    </Residue>
    <Residue name="CGLN">
      <Atom charge="-0.3821" name="N" type="N238"/>
      <Atom charge="0.2681" name="H" type="H241"/>
      <Atom charge="-0.2248" name="CA" type="C283"/>
      <Atom charge="0.1232" name="HA" type="H140"/>
      <Atom charge="-0.0664" name="CB" type="C136"/>
      <Atom charge="0.0452" name="HB2" type="H140"/>
      <Atom charge="0.0452" name="HB3" type="H140"/>
      <Atom charge="-0.021" name="CG" type="C136"/>
      <Atom charge="0.0203" name="HG2" type="H140"/>
      <Atom charge="0.0203" name="HG3" type="H140"/>
      <Atom charge="0.7093" name="CD" type="C235"/>
      <Atom charge="-0.6098" name="OE1" type="O236"/>
      <Atom charge="-0.9574" name="NE2" type="N237"/>
      <Atom charge="0.4304" name="HE21" type="H240"/>
      <Atom charge="0.4304" name="HE22" type="H240"/>
      <Atom charge="0.7775" name="C" type="C271"/>
      <Atom charge="-0.8042" name="O" type="O272"/>
      <Atom charge="-0.8042" name="OXT" type="O272"/>
      <Bond atomName1="N" atomName2="H"/>
      <Bond atomName1="N" atomName2="CA"/>
      <Bond atomName1="CA" atomName2="HA"/>
      <Bond atomName1="CA" atomName2="CB"/>
      <Bond atomName1="CA" atomName2="C"/>
      <Bond atomName1="CB" atomName2="HB2"/>
      <Bond atomName1="CB" atomName2="HB3"/>
      <Bond atomName1="CB" atomName2="CG"/>
      <Bond atomName1="CG" atomName2="HG2"/>
      <Bond atomName1="CG" atomName2="HG3"/>
      <Bond atomName1="CG" atomName2="CD"/>
      <Bond atomName1="CD" atomName2="OE1"/>
      <Bond atomName1="CD" atomName2="NE2"/>
      <Bond atomName1="NE2" atomName2="HE21"/>
      <Bond atomName1="NE2" atomName2="HE22"/>
      <Bond atomName1="C" atomName2="O"/>
      <Bond atomName1="C" atomName2="OXT"/>
      <ExternalBond atomName="N"/>
    </Residue>
    <Residue name="CGLU">
      <Atom charge="-0.5192" name="N" type="N238"/>
      <Atom charge="0.3055" name="H" type="H241"/>
      <Atom charge="-0.2059" name="CA" type="C283"/>
      <Atom charge="0.1399" name="HA" type="H140"/>
      <Atom charge="0.0071" name="CB" type="C136"/>
      <Atom charge="-0.0078" name="HB2" type="H140"/>
      <Atom charge="-0.0078" name="HB3" type="H140"/>
      <Atom charge="0.0675" name="CG" type="C274"/>
      <Atom charge="-0.0548" name="HG2" type="H140"/>
      <Atom charge="-0.0548" name="HG3" type="H140"/>
      <Atom charge="0.8183" name="CD" type="C271"/>
      <Atom charge="-0.822" name="OE1" type="O272"/>
      <Atom charge="-0.822" name="OE2" type="O272"/>
      <Atom charge="0.742" name="C" type="C271"/>
      <Atom charge="-0.793" name="O" type="O272"/>
      <Atom charge="-0.793" name="OXT" type="O272"/>
      <Bond atomName1="N" atomName2="H"/>
      <Bond atomName1="N" atomName2="CA"/>
      <Bond atomName1="CA" atomName2="HA"/>
      <Bond atomName1="CA" atomName2="CB"/>
      <Bond atomName1="CA" atomName2="C"/>
      <Bond atomName1="CB" atomName2="HB2"/>
      <Bond atomName1="CB" atomName2="HB3"/>
      <Bond atomName1="CB" atomName2="CG"/>
      <Bond atomName1="CG" atomName2="HG2"/>
      <Bond atomName1="CG" atomName2="HG3"/>
      <Bond atomName1="CG" atomName2="CD"/>
      <Bond atomName1="CD" atomName2="OE1"/>
      <Bond atomName1="CD" atomName2="OE2"/>
      <Bond atomName1="C" atomName2="O"/>
      <Bond atomName1="C" atomName2="OXT"/>
      <ExternalBond atomName="N"/>
    </Residue>
    <Residue name="CGLY">
      <Atom charge="-0.3821" name="N" type="N238"/>
      <Atom charge="0.2681" name="H" type="H241"/>
      <Atom charge="-0.2493" name="CA" type="C284"/>
      <Atom charge="0.1056" name="HA2" type="H140"/>
      <Atom charge="0.1056" name="HA3" type="H140"/>
      <Atom charge="0.7231" name="C" type="C271"/>
      <Atom charge="-0.7855" name="O" type="O272"/>
      <Atom charge="-0.7855" name="OXT" type="O272"/>
      <Bond atomName1="N" atomName2="H"/>
      <Bond atomName1="N" atomName2="CA"/>
      <Bond atomName1="CA" atomName2="HA2"/>
      <Bond atomName1="CA" atomName2="HA3"/>
      <Bond atomName1="CA" atomName2="C"/>
      <Bond atomName1="C" atomName2="O"/>
      <Bond atomName1="C" atomName2="OXT"/>
      <ExternalBond atomName="N"/>
    </Residue>
    <Residue name="CHID">
      <Atom charge="-0.3821" name="N" type="N238"/>
      <Atom charge="0.2681" name="H" type="H241"/>
      <Atom charge="-0.1739" name="CA" type="C283"/>
      <Atom charge="0.11" name="HA" type="H140"/>
      <Atom charge="-0.1046" name="CB" type="C505"/>
      <Atom charge="0.0565" name="HB2" type="H140"/>
      <Atom charge="0.0565" name="HB3" type="H140"/>
      <Atom charge="0.0293" name="CG" type="C508"/>
      <Atom charge="-0.3892" name="ND1" type="N503"/>
      <Atom charge="0.3755" name="HD1" type="H504"/>
      <Atom charge="0.1925" name="CE1" type="C506"/>
      <Atom charge="0.1418" name="HE1" type="H146"/>
      <Atom charge="-0.5629" name="NE2" type="N511"/>
      <Atom charge="0.1001" name="CD2" type="C507"/>
      <Atom charge="0.1241" name="HD2" type="H146"/>
      <Atom charge="0.7615" name="C" type="C271"/>
      <Atom charge="-0.8016" name="O" type="O272"/>
      <Atom charge="-0.8016" name="OXT" type="O272"/>
      <Bond atomName1="N" atomName2="H"/>
      <Bond atomName1="N" atomName2="CA"/>
      <Bond atomName1="CA" atomName2="HA"/>
      <Bond atomName1="CA" atomName2="CB"/>
      <Bond atomName1="CA" atomName2="C"/>
      <Bond atomName1="CB" atomName2="HB2"/>
      <Bond atomName1="CB" atomName2="HB3"/>
      <Bond atomName1="CB" atomName2="CG"/>
      <Bond atomName1="CG" atomName2="ND1"/>
      <Bond atomName1="CG" atomName2="CD2"/>
      <Bond atomName1="ND1" atomName2="HD1"/>
      <Bond atomName1="ND1" atomName2="CE1"/>
      <Bond atomName1="CE1" atomName2="HE1"/>
      <Bond atomName1="CE1" atomName2="NE2"/>
      <Bond atomName1="NE2" atomName2="CD2"/>
      <Bond atomName1="CD2" atomName2="HD2"/>
      <Bond atomName1="C" atomName2="O"/>
      <Bond atomName1="C" atomName2="OXT"/>
      <ExternalBond atomName="N"/>
    </Residue>
    <Residue name="CHIE">
      <Atom charge="-0.3821" name="N" type="N238"/>
      <Atom charge="0.2681" name="H" type="H241"/>
      <Atom charge="-0.2699" name="CA" type="C283"/>
      <Atom charge="0.165" name="HA" type="H140"/>
      <Atom charge="-0.1068" name="CB" type="C505"/>
      <Atom charge="0.062" name="HB2" type="H140"/>
      <Atom charge="0.062" name="HB3" type="H140"/>
      <Atom charge="0.2724" name="CG" type="C507"/>
      <Atom charge="-0.5517" name="ND1" type="N511"/>
      <Atom charge="0.1558" name="CE1" type="C506"/>
      <Atom charge="0.1448" name="HE1" type="H146"/>
      <Atom charge="-0.267" name="NE2" type="N503"/>
      <Atom charge="0.3319" name="HE2" type="H504"/>
      <Atom charge="-0.2588" name="CD2" type="C508"/>
      <Atom charge="0.1957" name="HD2" type="H146"/>
      <Atom charge="0.7916" name="C" type="C271"/>
      <Atom charge="-0.8065" name="O" type="O272"/>
      <Atom charge="-0.8065" name="OXT" type="O272"/>
      <Bond atomName1="N" atomName2="H"/>
      <Bond atomName1="N" atomName2="CA"/>
      <Bond atomName1="CA" atomName2="HA"/>
      <Bond atomName1="CA" atomName2="CB"/>
      <Bond atomName1="CA" atomName2="C"/>
      <Bond atomName1="CB" atomName2="HB2"/>
      <Bond atomName1="CB" atomName2="HB3"/>
      <Bond atomName1="CB" atomName2="CG"/>
      <Bond atomName1="CG" atomName2="ND1"/>
      <Bond atomName1="CG" atomName2="CD2"/>
      <Bond atomName1="ND1" atomName2="CE1"/>
      <Bond atomName1="CE1" atomName2="HE1"/>
      <Bond atomName1="CE1" atomName2="NE2"/>
      <Bond atomName1="NE2" atomName2="HE2"/>
      <Bond atomName1="NE2" atomName2="CD2"/>
      <Bond atomName1="CD2" atomName2="HD2"/>
      <Bond atomName1="C" atomName2="O"/>
      <Bond atomName1="C" atomName2="OXT"/>
      <ExternalBond atomName="N"/>
    </Residue>
    <Residue name="CHIP">
      <Atom charge="-0.3481" name="N" type="N238"/>
      <Atom charge="0.2764" name="H" type="H241"/>
      <Atom charge="-0.1445" name="CA" type="C283"/>
      <Atom charge="0.1115" name="HA" type="H140"/>
      <Atom charge="-0.08" name="CB" type="C505"/>
      <Atom charge="0.0868" name="HB2" type="H140"/>
      <Atom charge="0.0868" name="HB3" type="H140"/>
      <Atom charge="0.0298" name="CG" type="C510"/>
      <Atom charge="-0.1501" name="ND1" type="N512"/>
      <Atom charge="0.3883" name="HD1" type="H513"/>
      <Atom charge="-0.0251" name="CE1" type="C509"/>
      <Atom charge="0.2694" name="HE1" type="H146"/>
      <Atom charge="-0.1683" name="NE2" type="N512"/>
      <Atom charge="0.3913" name="HE2" type="H513"/>
      <Atom charge="-0.1256" name="CD2" type="C510"/>
      <Atom charge="0.2336" name="HD2" type="H146"/>
      <Atom charge="0.8032" name="C" type="C271"/>
      <Atom charge="-0.8177" name="O" type="O272"/>
      <Atom charge="-0.8177" name="OXT" type="O272"/>
      <Bond atomName1="N" atomName2="H"/>
      <Bond atomName1="N" atomName2="CA"/>
      <Bond atomName1="CA" atomName2="HA"/>
      <Bond atomName1="CA" atomName2="CB"/>
      <Bond atomName1="CA" atomName2="C"/>
      <Bond atomName1="CB" atomName2="HB2"/>
      <Bond atomName1="CB" atomName2="HB3"/>
      <Bond atomName1="CB" atomName2="CG"/>
      <Bond atomName1="CG" atomName2="ND1"/>
      <Bond atomName1="CG" atomName2="CD2"/>
      <Bond atomName1="ND1" atomName2="HD1"/>
      <Bond atomName1="ND1" atomName2="CE1"/>
      <Bond atomName1="CE1" atomName2="HE1"/>
      <Bond atomName1="CE1" atomName2="NE2"/>
      <Bond atomName1="NE2" atomName2="HE2"/>
      <Bond atomName1="NE2" atomName2="CD2"/>
      <Bond atomName1="CD2" atomName2="HD2"/>
      <Bond atomName1="C" atomName2="O"/>
      <Bond atomName1="C" atomName2="OXT"/>
      <ExternalBond atomName="N"/>
    </Residue>
    <Residue name="CILE">
      <Atom charge="-0.3821" name="N" type="N238"/>
      <Atom charge="0.2681" name="H" type="H241"/>
      <Atom charge="-0.31" name="CA" type="C283"/>
      <Atom charge="0.1375" name="HA" type="H140"/>
      <Atom charge="0.0363" name="CB" type="C137"/>
      <Atom charge="0.0766" name="HB" type="H140"/>
      <Atom charge="-0.3498" name="CG2" type="C135"/>
      <Atom charge="0.1021" name="HG21" type="H140"/>
      <Atom charge="0.1021" name="HG22" type="H140"/>
      <Atom charge="0.1021" name="HG23" type="H140"/>
      <Atom charge="-0.0323" name="CG1" type="C136"/>
      <Atom charge="0.0321" name="HG12" type="H140"/>
      <Atom charge="0.0321" name="HG13" type="H140"/>
      <Atom charge="-0.0699" name="CD1" type="C135"/>
      <Atom charge="0.0196" name="HD11" type="H140"/>
      <Atom charge="0.0196" name="HD12" type="H140"/>
      <Atom charge="0.0196" name="HD13" type="H140"/>
      <Atom charge="0.8343" name="C" type="C271"/>
      <Atom charge="-0.819" name="O" type="O272"/>
      <Atom charge="-0.819" name="OXT" type="O272"/>
      <Bond atomName1="N" atomName2="H"/>
      <Bond atomName1="N" atomName2="CA"/>
      <Bond atomName1="CA" atomName2="HA"/>
      <Bond atomName1="CA" atomName2="CB"/>
      <Bond atomName1="CA" atomName2="C"/>
      <Bond atomName1="CB" atomName2="HB"/>
      <Bond atomName1="CB" atomName2="CG2"/>
      <Bond atomName1="CB" atomName2="CG1"/>
      <Bond atomName1="CG2" atomName2="HG21"/>
      <Bond atomName1="CG2" atomName2="HG22"/>
      <Bond atomName1="CG2" atomName2="HG23"/>
      <Bond atomName1="CG1" atomName2="HG12"/>
      <Bond atomName1="CG1" atomName2="HG13"/>
      <Bond atomName1="CG1" atomName2="CD1"/>
      <Bond atomName1="CD1" atomName2="HD11"/>
      <Bond atomName1="CD1" atomName2="HD12"/>
      <Bond atomName1="CD1" atomName2="HD13"/>
      <Bond atomName1="C" atomName2="O"/>
      <Bond atomName1="C" atomName2="OXT"/>
      <ExternalBond atomName="N"/>
    </Residue>
    <Residue name="CLEU">
      <Atom charge="-0.3821" name="N" type="N238"/>
      <Atom charge="0.2681" name="H" type="H241"/>
      <Atom charge="-0.2847" name="CA" type="C283"/>
      <Atom charge="0.1346" name="HA" type="H140"/>
      <Atom charge="-0.2469" name="CB" type="C136"/>
      <Atom charge="0.0974" name="HB2" type="H140"/>
      <Atom charge="0.0974" name="HB3" type="H140"/>
      <Atom charge="0.3706" name="CG" type="C137"/>
      <Atom charge="-0.0374" name="HG" type="H140"/>
      <Atom charge="-0.4163" name="CD1" type="C135"/>
      <Atom charge="0.1038" name="HD11" type="H140"/>
      <Atom charge="0.1038" name="HD12" type="H140"/>
      <Atom charge="0.1038" name="HD13" type="H140"/>
      <Atom charge="-0.4163" name="CD2" type="C135"/>
      <Atom charge="0.1038" name="HD21" type="H140"/>
      <Atom charge="0.1038" name="HD22" type="H140"/>
      <Atom charge="0.1038" name="HD23" type="H140"/>
      <Atom charge="0.8326" name="C" type="C271"/>
      <Atom charge="-0.8199" name="O" type="O272"/>
      <Atom charge="-0.8199" name="OXT" type="O272"/>
      <Bond atomName1="N" atomName2="H"/>
      <Bond atomName1="N" atomName2="CA"/>
      <Bond atomName1="CA" atomName2="HA"/>
      <Bond atomName1="CA" atomName2="CB"/>
      <Bond atomName1="CA" atomName2="C"/>
      <Bond atomName1="CB" atomName2="HB2"/>
      <Bond atomName1="CB" atomName2="HB3"/>
      <Bond atomName1="CB" atomName2="CG"/>
      <Bond atomName1="CG" atomName2="HG"/>
      <Bond atomName1="CG" atomName2="CD1"/>
      <Bond atomName1="CG" atomName2="CD2"/>
      <Bond atomName1="CD1" atomName2="HD11"/>
      <Bond atomName1="CD1" atomName2="HD12"/>
      <Bond atomName1="CD1" atomName2="HD13"/>
      <Bond atomName1="CD2" atomName2="HD21"/>
      <Bond atomName1="CD2" atomName2="HD22"/>
      <Bond atomName1="CD2" atomName2="HD23"/>
      <Bond atomName1="C" atomName2="O"/>
      <Bond atomName1="C" atomName2="OXT"/>
      <ExternalBond atomName="N"/>
    </Residue>
    <Residue name="CLYS">
      <Atom charge="-0.3481" name="N" type="N238"/>
      <Atom charge="0.2764" name="H" type="H241"/>
      <Atom charge="-0.2903" name="CA" type="C283"/>
      <Atom charge="0.1438" name="HA" type="H140"/>
      <Atom charge="-0.0538" name="CB" type="C136"/>
      <Atom charge="0.0482" name="HB2" type="H140"/>
      <Atom charge="0.0482" name="HB3" type="H140"/>
      <Atom charge="0.0227" name="CG" type="C136"/>
      <Atom charge="0.0134" name="HG2" type="H140"/>
      <Atom charge="0.0134" name="HG3" type="H140"/>
      <Atom charge="-0.0392" name="CD" type="C136"/>
      <Atom charge="0.0611" name="HD2" type="H140"/>
      <Atom charge="0.0611" name="HD3" type="H140"/>
      <Atom charge="-0.0176" name="CE" type="C292"/>
      <Atom charge="0.1121" name="HE2" type="H140"/>
      <Atom charge="0.1121" name="HE3" type="H140"/>
      <Atom charge="-0.3741" name="NZ" type="N287"/>
      <Atom charge="0.3374" name="HZ1" type="H290"/>
      <Atom charge="0.3374" name="HZ2" type="H290"/>
      <Atom charge="0.3374" name="HZ3" type="H290"/>
      <Atom charge="0.8488" name="C" type="C271"/>
      <Atom charge="-0.8252" name="O" type="O272"/>
      <Atom charge="-0.8252" name="OXT" type="O272"/>
      <Bond atomName1="N" atomName2="H"/>
      <Bond atomName1="N" atomName2="CA"/>
      <Bond atomName1="CA" atomName2="HA"/>
      <Bond atomName1="CA" atomName2="CB"/>
      <Bond atomName1="CA" atomName2="C"/>
      <Bond atomName1="CB" atomName2="HB2"/>
      <Bond atomName1="CB" atomName2="HB3"/>
      <Bond atomName1="CB" atomName2="CG"/>
      <Bond atomName1="CG" atomName2="HG2"/>
      <Bond atomName1="CG" atomName2="HG3"/>
      <Bond atomName1="CG" atomName2="CD"/>
      <Bond atomName1="CD" atomName2="HD2"/>
      <Bond atomName1="CD" atomName2="HD3"/>
      <Bond atomName1="CD" atomName2="CE"/>
      <Bond atomName1="CE" atomName2="HE2"/>
      <Bond atomName1="CE" atomName2="HE3"/>
      <Bond atomName1="CE" atomName2="NZ"/>
      <Bond atomName1="NZ" atomName2="HZ1"/>
      <Bond atomName1="NZ" atomName2="HZ2"/>
      <Bond atomName1="NZ" atomName2="HZ3"/>
      <Bond atomName1="C" atomName2="O"/>
      <Bond atomName1="C" atomName2="OXT"/>
      <ExternalBond atomName="N"/>
    </Residue>
    <Residue name="CMET">
      <Atom charge="-0.3821" name="N" type="N238"/>
      <Atom charge="0.2681" name="H" type="H241"/>
      <Atom charge="-0.2597" name="CA" type="C283"/>
      <Atom charge="0.1277" name="HA" type="H140"/>
      <Atom charge="-0.0236" name="CB" type="C136"/>
      <Atom charge="0.048" name="HB2" type="H140"/>
      <Atom charge="0.048" name="HB3" type="H140"/>
      <Atom charge="0.0492" name="CG" type="C210"/>
      <Atom charge="0.0317" name="HG2" type="H140"/>
      <Atom charge="0.0317" name="HG3" type="H140"/>
      <Atom charge="-0.2692" name="SD" type="S202"/>
      <Atom charge="-0.0376" name="CE" type="C209"/>
      <Atom charge="0.0625" name="HE1" type="H140"/>
      <Atom charge="0.0625" name="HE2" type="H140"/>
      <Atom charge="0.0625" name="HE3" type="H140"/>
      <Atom charge="0.8013" name="C" type="C271"/>
      <Atom charge="-0.8105" name="O" type="O272"/>
      <Atom charge="-0.8105" name="OXT" type="O272"/>
      <Bond atomName1="N" atomName2="H"/>
      <Bond atomName1="N" atomName2="CA"/>
      <Bond atomName1="CA" atomName2="HA"/>
      <Bond atomName1="CA" atomName2="CB"/>
      <Bond atomName1="CA" atomName2="C"/>
      <Bond atomName1="CB" atomName2="HB2"/>
      <Bond atomName1="CB" atomName2="HB3"/>
      <Bond atomName1="CB" atomName2="CG"/>
      <Bond atomName1="CG" atomName2="HG2"/>
      <Bond atomName1="CG" atomName2="HG3"/>
      <Bond atomName1="CG" atomName2="SD"/>
      <Bond atomName1="SD" atomName2="CE"/>
      <Bond atomName1="CE" atomName2="HE1"/>
      <Bond atomName1="CE" atomName2="HE2"/>
      <Bond atomName1="CE" atomName2="HE3"/>
      <Bond atomName1="C" atomName2="O"/>
      <Bond atomName1="C" atomName2="OXT"/>
      <ExternalBond atomName="N"/>
    </Residue>
    <Residue name="CPHE">
      <Atom charge="-0.3821" name="N" type="N238"/>
      <Atom charge="0.2681" name="H" type="H241"/>
      <Atom charge="-0.1825" name="CA" type="C283"/>
      <Atom charge="0.1098" name="HA" type="H140"/>
      <Atom charge="-0.0959" name="CB" type="C149"/>
      <Atom charge="0.0443" name="HB2" type="H140"/>
      <Atom charge="0.0443" name="HB3" type="H140"/>
      <Atom charge="0.0552" name="CG" type="C145"/>
      <Atom charge="-0.13" name="CD1" type="C145"/>
      <Atom charge="0.1408" name="HD1" type="H146"/>
      <Atom charge="-0.1847" name="CE1" type="C145"/>
      <Atom charge="0.1461" name="HE1" type="H146"/>
      <Atom charge="-0.0944" name="CZ" type="C145"/>
      <Atom charge="0.128" name="HZ" type="H146"/>
      <Atom charge="-0.1847" name="CE2" type="C145"/>
      <Atom charge="0.1461" name="HE2" type="H146"/>
      <Atom charge="-0.13" name="CD2" type="C145"/>
      <Atom charge="0.1408" name="HD2" type="H146"/>
      <Atom charge="0.766" name="C" type="C271"/>
      <Atom charge="-0.8026" name="O" type="O272"/>
      <Atom charge="-0.8026" name="OXT" type="O272"/>
      <Bond atomName1="N" atomName2="H"/>
      <Bond atomName1="N" atomName2="CA"/>
      <Bond atomName1="CA" atomName2="HA"/>
      <Bond atomName1="CA" atomName2="CB"/>
      <Bond atomName1="CA" atomName2="C"/>
      <Bond atomName1="CB" atomName2="HB2"/>
      <Bond atomName1="CB" atomName2="HB3"/>
      <Bond atomName1="CB" atomName2="CG"/>
      <Bond atomName1="CG" atomName2="CD1"/>
      <Bond atomName1="CG" atomName2="CD2"/>
      <Bond atomName1="CD1" atomName2="HD1"/>
      <Bond atomName1="CD1" atomName2="CE1"/>
      <Bond atomName1="CE1" atomName2="HE1"/>
      <Bond atomName1="CE1" atomName2="CZ"/>
      <Bond atomName1="CZ" atomName2="HZ"/>
      <Bond atomName1="CZ" atomName2="CE2"/>
      <Bond atomName1="CE2" atomName2="HE2"/>
      <Bond atomName1="CE2" atomName2="CD2"/>
      <Bond atomName1="CD2" atomName2="HD2"/>
      <Bond atomName1="C" atomName2="O"/>
      <Bond atomName1="C" atomName2="OXT"/>
      <ExternalBond atomName="N"/>
    </Residue>
    <Residue name="CPRO">
      <Atom charge="-0.2802" name="N" type="N239"/>
      <Atom charge="0.0434" name="CD" type="C245"/>
      <Atom charge="0.0331" name="HD2" type="H140"/>
      <Atom charge="0.0331" name="HD3" type="H140"/>
      <Atom charge="0.0466" name="CG" type="C136"/>
      <Atom charge="0.0172" name="HG2" type="H140"/>
      <Atom charge="0.0172" name="HG3" type="H140"/>
      <Atom charge="-0.0543" name="CB" type="C136"/>
      <Atom charge="0.0381" name="HB2" type="H140"/>
      <Atom charge="0.0381" name="HB3" type="H140"/>
      <Atom charge="-0.1336" name="CA" type="C285"/>
      <Atom charge="0.0776" name="HA" type="H140"/>
      <Atom charge="0.6631" name="C" type="C271"/>
      <Atom charge="-0.7697" name="O" type="O272"/>
      <Atom charge="-0.7697" name="OXT" type="O272"/>
      <Bond atomName1="N" atomName2="CD"/>
      <Bond atomName1="N" atomName2="CA"/>
      <Bond atomName1="CD" atomName2="HD2"/>
      <Bond atomName1="CD" atomName2="HD3"/>
      <Bond atomName1="CD" atomName2="CG"/>
      <Bond atomName1="CG" atomName2="HG2"/>
      <Bond atomName1="CG" atomName2="HG3"/>
      <Bond atomName1="CG" atomName2="CB"/>
      <Bond atomName1="CB" atomName2="HB2"/>
      <Bond atomName1="CB" atomName2="HB3"/>
      <Bond atomName1="CB" atomName2="CA"/>
      <Bond atomName1="CA" atomName2="HA"/>
      <Bond atomName1="CA" atomName2="C"/>
      <Bond atomName1="C" atomName2="O"/>
      <Bond atomName1="C" atomName2="OXT"/>
      <ExternalBond atomName="N"/>
    </Residue>
    <Residue name="CSER">
      <Atom charge="-0.3821" name="N" type="N238"/>
      <Atom charge="0.2681" name="H" type="H241"/>
      <Atom charge="-0.2722" name="CA" type="C283"/>
      <Atom charge="0.1304" name="HA" type="H140"/>
      <Atom charge="0.1123" name="CB" type="C157"/>
      <Atom charge="0.0813" name="HB2" type="H140"/>
      <Atom charge="0.0813" name="HB3" type="H140"/>
      <Atom charge="-0.6514" name="OG" type="O154"/>
      <Atom charge="0.4474" name="HG" type="H155"/>
      <Atom charge="0.8113" name="C" type="C271"/>
      <Atom charge="-0.8132" name="O" type="O272"/>
      <Atom charge="-0.8132" name="OXT" type="O272"/>
      <Bond atomName1="N" atomName2="H"/>
      <Bond atomName1="N" atomName2="CA"/>
      <Bond atomName1="CA" atomName2="HA"/>
      <Bond atomName1="CA" atomName2="CB"/>
      <Bond atomName1="CA" atomName2="C"/>
      <Bond atomName1="CB" atomName2="HB2"/>
      <Bond atomName1="CB" atomName2="HB3"/>
      <Bond atomName1="CB" atomName2="OG"/>
      <Bond atomName1="OG" atomName2="HG"/>
      <Bond atomName1="C" atomName2="O"/>
      <Bond atomName1="C" atomName2="OXT"/>
      <ExternalBond atomName="N"/>
    </Residue>
    <Residue name="CTHR">
      <Atom charge="-0.3821" name="N" type="N238"/>
      <Atom charge="0.2681" name="H" type="H241"/>
      <Atom charge="-0.242" name="CA" type="C283"/>
      <Atom charge="0.1207" name="HA" type="H140"/>
      <Atom charge="0.3025" name="CB" type="C158"/>
      <Atom charge="0.0078" name="HB" type="H140"/>
      <Atom charge="-0.1853" name="CG2" type="C135"/>
      <Atom charge="0.0586" name="HG21" type="H140"/>
      <Atom charge="0.0586" name="HG22" type="H140"/>
      <Atom charge="0.0586" name="HG23" type="H140"/>
      <Atom charge="-0.6496" name="OG1" type="O154"/>
      <Atom charge="0.4119" name="HG1" type="H155"/>
      <Atom charge="0.781" name="C" type="C271"/>
      <Atom charge="-0.8044" name="O" type="O272"/>
      <Atom charge="-0.8044" name="OXT" type="O272"/>
      <Bond atomName1="N" atomName2="H"/>
      <Bond atomName1="N" atomName2="CA"/>
      <Bond atomName1="CA" atomName2="HA"/>
      <Bond atomName1="CA" atomName2="CB"/>
      <Bond atomName1="CA" atomName2="C"/>
      <Bond atomName1="CB" atomName2="HB"/>
      <Bond atomName1="CB" atomName2="CG2"/>
      <Bond atomName1="CB" atomName2="OG1"/>
      <Bond atomName1="CG2" atomName2="HG21"/>
      <Bond atomName1="CG2" atomName2="HG22"/>
      <Bond atomName1="CG2" atomName2="HG23"/>
      <Bond atomName1="OG1" atomName2="HG1"/>
      <Bond atomName1="C" atomName2="O"/>
      <Bond atomName1="C" atomName2="OXT"/>
      <ExternalBond atomName="N"/>
    </Residue>
    <Residue name="CTRP">
      <Atom charge="-0.3821" name="N" type="N238"/>
      <Atom charge="0.2681" name="H" type="H241"/>
      <Atom charge="-0.2084" name="CA" type="C283"/>
      <Atom charge="0.1272" name="HA" type="H140"/>
      <Atom charge="-0.0742" name="CB" type="C136"/>
      <Atom charge="0.0497" name="HB2" type="H140"/>
      <Atom charge="0.0497" name="HB3" type="H140"/>
      <Atom charge="-0.0796" name="CG" type="C500"/>
      <Atom charge="-0.1808" name="CD1" type="C514"/>
      <Atom charge="0.2043" name="HD1" type="H146"/>
      <Atom charge="-0.3316" name="NE1" type="N503"/>
      <Atom charge="0.3413" name="HE1" type="H504"/>
      <Atom charge="0.1222" name="CE2" type="C502"/>
      <Atom charge="-0.2594" name="CZ2" type="C145"/>
      <Atom charge="0.1567" name="HZ2" type="H146"/>
      <Atom charge="-0.102" name="CH2" type="C145"/>
      <Atom charge="0.1401" name="HH2" type="H146"/>
      <Atom charge="-0.2287" name="CZ3" type="C145"/>
      <Atom charge="0.1507" name="HZ3" type="H146"/>
      <Atom charge="-0.1837" name="CE3" type="C145"/>
      <Atom charge="0.1491" name="HE3" type="H146"/>
      <Atom charge="0.1078" name="CD2" type="C501"/>
      <Atom charge="0.7658" name="C" type="C271"/>
      <Atom charge="-0.8011" name="O" type="O272"/>
      <Atom charge="-0.8011" name="OXT" type="O272"/>
      <Bond atomName1="N" atomName2="H"/>
      <Bond atomName1="N" atomName2="CA"/>
      <Bond atomName1="CA" atomName2="HA"/>
      <Bond atomName1="CA" atomName2="CB"/>
      <Bond atomName1="CA" atomName2="C"/>
      <Bond atomName1="CB" atomName2="HB2"/>
      <Bond atomName1="CB" atomName2="HB3"/>
      <Bond atomName1="CB" atomName2="CG"/>
      <Bond atomName1="CG" atomName2="CD1"/>
      <Bond atomName1="CG" atomName2="CD2"/>
      <Bond atomName1="CD1" atomName2="HD1"/>
      <Bond atomName1="CD1" atomName2="NE1"/>
      <Bond atomName1="NE1" atomName2="HE1"/>
      <Bond atomName1="NE1" atomName2="CE2"/>
      <Bond atomName1="CE2" atomName2="CZ2"/>
      <Bond atomName1="CE2" atomName2="CD2"/>
      <Bond atomName1="CZ2" atomName2="HZ2"/>
      <Bond atomName1="CZ2" atomName2="CH2"/>
      <Bond atomName1="CH2" atomName2="HH2"/>
      <Bond atomName1="CH2" atomName2="CZ3"/>
      <Bond atomName1="CZ3" atomName2="HZ3"/>
      <Bond atomName1="CZ3" atomName2="CE3"/>
      <Bond atomName1="CE3" atomName2="HE3"/>
      <Bond atomName1="CE3" atomName2="CD2"/>
      <Bond atomName1="C" atomName2="O"/>
      <Bond atomName1="C" atomName2="OXT"/>
      <ExternalBond atomName="N"/>
    </Residue>
    <Residue name="CTYR">
      <Atom charge="-0.3821" name="N" type="N238"/>
      <Atom charge="0.2681" name="H" type="H241"/>
      <Atom charge="-0.2015" name="CA" type="C283"/>
      <Atom charge="0.1092" name="HA" type="H140"/>
      <Atom charge="-0.0752" name="CB" type="C149"/>
      <Atom charge="0.049" name="HB2" type="H140"/>
      <Atom charge="0.049" name="HB3" type="H140"/>
      <Atom charge="0.0243" name="CG" type="C145"/>
      <Atom charge="-0.1922" name="CD1" type="C145"/>
      <Atom charge="0.178" name="HD1" type="H146"/>
      <Atom charge="-0.2458" name="CE1" type="C145"/>
      <Atom charge="0.1673" name="HE1" type="H146"/>
      <Atom charge="0.3395" name="CZ" type="C166"/>
      <Atom charge="-0.5643" name="OH" type="O167"/>
      <Atom charge="0.4017" name="HH" type="H168"/>
      <Atom charge="-0.2458" name="CE2" type="C145"/>
      <Atom charge="0.1673" name="HE2" type="H146"/>
      <Atom charge="-0.1922" name="CD2" type="C145"/>
      <Atom charge="0.178" name="HD2" type="H146"/>
      <Atom charge="0.7817" name="C" type="C271"/>
      <Atom charge="-0.807" name="O" type="O272"/>
      <Atom charge="-0.807" name="OXT" type="O272"/>
      <Bond atomName1="N" atomName2="H"/>
      <Bond atomName1="N" atomName2="CA"/>
      <Bond atomName1="CA" atomName2="HA"/>
      <Bond atomName1="CA" atomName2="CB"/>
      <Bond atomName1="CA" atomName2="C"/>
      <Bond atomName1="CB" atomName2="HB2"/>
      <Bond atomName1="CB" atomName2="HB3"/>
      <Bond atomName1="CB" atomName2="CG"/>
      <Bond atomName1="CG" atomName2="CD1"/>
      <Bond atomName1="CG" atomName2="CD2"/>
      <Bond atomName1="CD1" atomName2="HD1"/>
      <Bond atomName1="CD1" atomName2="CE1"/>
      <Bond atomName1="CE1" atomName2="HE1"/>
      <Bond atomName1="CE1" atomName2="CZ"/>
      <Bond atomName1="CZ" atomName2="OH"/>
      <Bond atomName1="CZ" atomName2="CE2"/>
      <Bond atomName1="OH" atomName2="HH"/>
      <Bond atomName1="CE2" atomName2="HE2"/>
      <Bond atomName1="CE2" atomName2="CD2"/>
      <Bond atomName1="CD2" atomName2="HD2"/>
      <Bond atomName1="C" atomName2="O"/>
      <Bond atomName1="C" atomName2="OXT"/>
      <ExternalBond atomName="N"/>
    </Residue>
    <Residue name="CVAL">
      <Atom charge="-0.3821" name="N" type="N238"/>
      <Atom charge="0.2681" name="H" type="H241"/>
      <Atom charge="-0.3438" name="CA" type="C283"/>
      <Atom charge="0.1438" name="HA" type="H140"/>
      <Atom charge="0.194" name="CB" type="C137"/>
      <Atom charge="0.0308" name="HB" type="H140"/>
      <Atom charge="-0.3064" name="CG1" type="C135"/>
      <Atom charge="0.0836" name="HG11" type="H140"/>
      <Atom charge="0.0836" name="HG12" type="H140"/>
      <Atom charge="0.0836" name="HG13" type="H140"/>
      <Atom charge="-0.3064" name="CG2" type="C135"/>
      <Atom charge="0.0836" name="HG21" type="H140"/>
      <Atom charge="0.0836" name="HG22" type="H140"/>
      <Atom charge="0.0836" name="HG23" type="H140"/>
      <Atom charge="0.835" name="C" type="C271"/>
      <Atom charge="-0.8173" name="O" type="O272"/>
      <Atom charge="-0.8173" name="OXT" type="O272"/>
      <Bond atomName1="N" atomName2="H"/>
      <Bond atomName1="N" atomName2="CA"/>
      <Bond atomName1="CA" atomName2="HA"/>
      <Bond atomName1="CA" atomName2="CB"/>
      <Bond atomName1="CA" atomName2="C"/>
      <Bond atomName1="CB" atomName2="HB"/>
      <Bond atomName1="CB" atomName2="CG1"/>
      <Bond atomName1="CB" atomName2="CG2"/>
      <Bond atomName1="CG1" atomName2="HG11"/>
      <Bond atomName1="CG1" atomName2="HG12"/>
      <Bond atomName1="CG1" atomName2="HG13"/>
      <Bond atomName1="CG2" atomName2="HG21"/>
      <Bond atomName1="CG2" atomName2="HG22"/>
      <Bond atomName1="CG2" atomName2="HG23"/>
      <Bond atomName1="C" atomName2="O"/>
      <Bond atomName1="C" atomName2="OXT"/>
      <ExternalBond atomName="N"/>
    </Residue>
    <Residue name="NHE">
      <Atom charge="-0.463" name="N" type="N237"/>
      <Atom charge="0.2315" name="HN1" type="H240"/>
      <Atom charge="0.2315" name="HN2" type="H240"/>
      <Bond atomName1="N" atomName2="HN1"/>
      <Bond atomName1="N" atomName2="HN2"/>
      <ExternalBond atomName="N"/>
    </Residue>
    <Residue name="NME">
      <Atom charge="-0.4412" name="N" type="N238"/>
      <Atom charge="0.3083" name="H" type="H241"/>
      <Atom charge="-0.0041" name="CH3" type="C242"/>
      <Atom charge="0.0613" name="HH31" type="H140"/>
      <Atom charge="0.0613" name="HH32" type="H140"/>
      <Atom charge="0.0613" name="HH33" type="H140"/>
      <Bond atomName1="N" atomName2="H"/>
      <Bond atomName1="N" atomName2="CH3"/>
      <Bond atomName1="CH3" atomName2="HH31"/>
      <Bond atomName1="CH3" atomName2="HH32"/>
      <Bond atomName1="CH3" atomName2="HH33"/>
      <ExternalBond atomName="N"/>
    </Residue>
    <Residue name="ACE">
      <Atom charge="0.1504" name="H1" type="H140"/>
      <Atom charge="-0.5264" name="CH3" type="C135"/>
      <Atom charge="0.1504" name="H2" type="H140"/>
      <Atom charge="0.1504" name="H3" type="H140"/>
      <Atom charge="0.7857" name="C" type="C235"/>
      <Atom charge="-0.6082" name="O" type="O236"/>
      <Bond atomName1="H1" atomName2="CH3"/>
      <Bond atomName1="CH3" atomName2="H2"/>
      <Bond atomName1="CH3" atomName2="H3"/>
      <Bond atomName1="CH3" atomName2="C"/>
      <Bond atomName1="C" atomName2="O"/>
      <ExternalBond atomName="C"/>
    </Residue>
    <Residue name="NALA">
      <Atom charge="0.1414" name="N" type="N287"/>
      <Atom charge="0.1997" name="H1" type="H290"/>
      <Atom charge="0.1997" name="H2" type="H290"/>
      <Atom charge="0.1997" name="H3" type="H290"/>
      <Atom charge="0.0962" name="CA" type="C293"/>
      <Atom charge="0.0889" name="HA" type="H140"/>
      <Atom charge="-0.0597" name="CB" type="C135"/>
      <Atom charge="0.03" name="HB1" type="H140"/>
      <Atom charge="0.03" name="HB2" type="H140"/>
      <Atom charge="0.03" name="HB3" type="H140"/>
      <Atom charge="0.6163" name="C" type="C235"/>
      <Atom charge="-0.5722" name="O" type="O236"/>
      <Bond atomName1="N" atomName2="H1"/>
      <Bond atomName1="N" atomName2="H2"/>
      <Bond atomName1="N" atomName2="H3"/>
      <Bond atomName1="N" atomName2="CA"/>
      <Bond atomName1="CA" atomName2="HA"/>
      <Bond atomName1="CA" atomName2="CB"/>
      <Bond atomName1="CA" atomName2="C"/>
      <Bond atomName1="CB" atomName2="HB1"/>
      <Bond atomName1="CB" atomName2="HB2"/>
      <Bond atomName1="CB" atomName2="HB3"/>
      <Bond atomName1="C" atomName2="O"/>
      <ExternalBond atomName="C"/>
    </Residue>
    <Residue name="NARG">
      <Atom charge="0.1305" name="N" type="N287"/>
      <Atom charge="0.2083" name="H1" type="H290"/>
      <Atom charge="0.2083" name="H2" type="H290"/>
      <Atom charge="0.2083" name="H3" type="H290"/>
      <Atom charge="-0.0223" name="CA" type="C293"/>
      <Atom charge="0.1242" name="HA" type="H140"/>
      <Atom charge="0.0118" name="CB" type="C136"/>
      <Atom charge="0.0226" name="HB2" type="H140"/>
      <Atom charge="0.0226" name="HB3" type="H140"/>
      <Atom charge="0.0236" name="CG" type="C308"/>
      <Atom charge="0.0309" name="HG2" type="H140"/>
      <Atom charge="0.0309" name="HG3" type="H140"/>
      <Atom charge="0.0935" name="CD" type="C307"/>
      <Atom charge="0.0527" name="HD2" type="H140"/>
      <Atom charge="0.0527" name="HD3" type="H140"/>
      <Atom charge="-0.565" name="NE" type="N303"/>
      <Atom charge="0.3592" name="HE" type="H304"/>
      <Atom charge="0.8281" name="CZ" type="C302"/>
      <Atom charge="-0.8693" name="NH1" type="N300"/>
      <Atom charge="0.4494" name="HH11" type="H301"/>
      <Atom charge="0.4494" name="HH12" type="H301"/>
      <Atom charge="-0.8693" name="NH2" type="N300"/>
      <Atom charge="0.4494" name="HH21" type="H301"/>
      <Atom charge="0.4494" name="HH22" type="H301"/>
      <Atom charge="0.7214" name="C" type="C235"/>
      <Atom charge="-0.6013" name="O" type="O236"/>
      <Bond atomName1="N" atomName2="H1"/>
      <Bond atomName1="N" atomName2="H2"/>
      <Bond atomName1="N" atomName2="H3"/>
      <Bond atomName1="N" atomName2="CA"/>
      <Bond atomName1="CA" atomName2="HA"/>
      <Bond atomName1="CA" atomName2="CB"/>
      <Bond atomName1="CA" atomName2="C"/>
      <Bond atomName1="CB" atomName2="HB2"/>
      <Bond atomName1="CB" atomName2="HB3"/>
      <Bond atomName1="CB" atomName2="CG"/>
      <Bond atomName1="CG" atomName2="HG2"/>
      <Bond atomName1="CG" atomName2="HG3"/>
      <Bond atomName1="CG" atomName2="CD"/>
      <Bond atomName1="CD" atomName2="HD2"/>
      <Bond atomName1="CD" atomName2="HD3"/>
      <Bond atomName1="CD" atomName2="NE"/>
      <Bond atomName1="NE" atomName2="HE"/>
      <Bond atomName1="NE" atomName2="CZ"/>
      <Bond atomName1="CZ" atomName2="NH1"/>
      <Bond atomName1="CZ" atomName2="NH2"/>
      <Bond atomName1="NH1" atomName2="HH11"/>
      <Bond atomName1="NH1" atomName2="HH12"/>
      <Bond atomName1="NH2" atomName2="HH21"/>
      <Bond atomName1="NH2" atomName2="HH22"/>
      <Bond atomName1="C" atomName2="O"/>
      <ExternalBond atomName="C"/>
    </Residue>
    <Residue name="NASN">
      <Atom charge="0.1801" name="N" type="N287"/>
      <Atom charge="0.1921" name="H1" type="H290"/>
      <Atom charge="0.1921" name="H2" type="H290"/>
      <Atom charge="0.1921" name="H3" type="H290"/>
      <Atom charge="0.0368" name="CA" type="C293"/>
      <Atom charge="0.1231" name="HA" type="H140"/>
      <Atom charge="-0.0283" name="CB" type="C136"/>
      <Atom charge="0.0515" name="HB2" type="H140"/>
      <Atom charge="0.0515" name="HB3" type="H140"/>
      <Atom charge="0.5833" name="CG" type="C235"/>
      <Atom charge="-0.5744" name="OD1" type="O236"/>
      <Atom charge="-0.8634" name="ND2" type="N237"/>
      <Atom charge="0.4097" name="HD21" type="H240"/>
      <Atom charge="0.4097" name="HD22" type="H240"/>
      <Atom charge="0.6163" name="C" type="C235"/>
      <Atom charge="-0.5722" name="O" type="O236"/>
      <Bond atomName1="N" atomName2="H1"/>
      <Bond atomName1="N" atomName2="H2"/>
      <Bond atomName1="N" atomName2="H3"/>
      <Bond atomName1="N" atomName2="CA"/>
      <Bond atomName1="CA" atomName2="HA"/>
      <Bond atomName1="CA" atomName2="CB"/>
      <Bond atomName1="CA" atomName2="C"/>
      <Bond atomName1="CB" atomName2="HB2"/>
      <Bond atomName1="CB" atomName2="HB3"/>
      <Bond atomName1="CB" atomName2="CG"/>
      <Bond atomName1="CG" atomName2="OD1"/>
      <Bond atomName1="CG" atomName2="ND2"/>
      <Bond atomName1="ND2" atomName2="HD21"/>
      <Bond atomName1="ND2" atomName2="HD22"/>
      <Bond atomName1="C" atomName2="O"/>
      <ExternalBond atomName="C"/>
    </Residue>
    <Residue name="NASP">
      <Atom charge="0.0782" name="N" type="N287"/>
      <Atom charge="0.22" name="H1" type="H290"/>
      <Atom charge="0.22" name="H2" type="H290"/>
      <Atom charge="0.22" name="H3" type="H290"/>
      <Atom charge="0.0292" name="CA" type="C293"/>
      <Atom charge="0.1141" name="HA" type="H140"/>
      <Atom charge="-0.0235" name="CB" type="C274"/>
      <Atom charge="-0.0169" name="HB2" type="H140"/>
      <Atom charge="-0.0169" name="HB3" type="H140"/>
      <Atom charge="0.8194" name="CG" type="C271"/>
      <Atom charge="-0.8084" name="OD1" type="O272"/>
      <Atom charge="-0.8084" name="OD2" type="O272"/>
      <Atom charge="0.5621" name="C" type="C235"/>
      <Atom charge="-0.5889" name="O" type="O236"/>
      <Bond atomName1="N" atomName2="H1"/>
      <Bond atomName1="N" atomName2="H2"/>
      <Bond atomName1="N" atomName2="H3"/>
      <Bond atomName1="N" atomName2="CA"/>
      <Bond atomName1="CA" atomName2="HA"/>
      <Bond atomName1="CA" atomName2="CB"/>
      <Bond atomName1="CA" atomName2="C"/>
      <Bond atomName1="CB" atomName2="HB2"/>
      <Bond atomName1="CB" atomName2="HB3"/>
      <Bond atomName1="CB" atomName2="CG"/>
      <Bond atomName1="CG" atomName2="OD1"/>
      <Bond atomName1="CG" atomName2="OD2"/>
      <Bond atomName1="C" atomName2="O"/>
      <ExternalBond atomName="C"/>
    </Residue>
    <Residue name="NCYS">
      <Atom charge="0.1325" name="N" type="N287"/>
      <Atom charge="0.2023" name="H1" type="H290"/>
      <Atom charge="0.2023" name="H2" type="H290"/>
      <Atom charge="0.2023" name="H3" type="H290"/>
      <Atom charge="0.0927" name="CA" type="C293"/>
      <Atom charge="0.1411" name="HA" type="H140"/>
      <Atom charge="-0.1195" name="CB" type="C206"/>
      <Atom charge="0.1188" name="HB2" type="H140"/>
      <Atom charge="0.1188" name="HB3" type="H140"/>
      <Atom charge="-0.3298" name="SG" type="S200"/>
      <Atom charge="0.1975" name="HG" type="H204"/>
      <Atom charge="0.6123" name="C" type="C235"/>
      <Atom charge="-0.5713" name="O" type="O236"/>
      <Bond atomName1="N" atomName2="H1"/>
      <Bond atomName1="N" atomName2="H2"/>
      <Bond atomName1="N" atomName2="H3"/>
      <Bond atomName1="N" atomName2="CA"/>
      <Bond atomName1="CA" atomName2="HA"/>
      <Bond atomName1="CA" atomName2="CB"/>
      <Bond atomName1="CA" atomName2="C"/>
      <Bond atomName1="CB" atomName2="HB2"/>
      <Bond atomName1="CB" atomName2="HB3"/>
      <Bond atomName1="CB" atomName2="SG"/>
      <Bond atomName1="SG" atomName2="HG"/>
      <Bond atomName1="C" atomName2="O"/>
      <ExternalBond atomName="C"/>
    </Residue>
    <Residue name="NCYX">
      <Atom charge="0.2069" name="N" type="N287"/>
      <Atom charge="0.1815" name="H1" type="H290"/>
      <Atom charge="0.1815" name="H2" type="H290"/>
      <Atom charge="0.1815" name="H3" type="H290"/>
      <Atom charge="0.1055" name="CA" type="C293"/>
      <Atom charge="0.0922" name="HA" type="H140"/>
      <Atom charge="-0.0277" name="CB" type="C214"/>
      <Atom charge="0.068" name="HB2" type="H140"/>
      <Atom charge="0.068" name="HB3" type="H140"/>
      <Atom charge="-0.0984" name="SG" type="S203"/>
      <Atom charge="0.6123" name="C" type="C235"/>
      <Atom charge="-0.5713" name="O" type="O236"/>
      <Bond atomName1="N" atomName2="H1"/>
      <Bond atomName1="N" atomName2="H2"/>
      <Bond atomName1="N" atomName2="H3"/>
      <Bond atomName1="N" atomName2="CA"/>
      <Bond atomName1="CA" atomName2="HA"/>
      <Bond atomName1="CA" atomName2="CB"/>
      <Bond atomName1="CA" atomName2="C"/>
      <Bond atomName1="CB" atomName2="HB2"/>
      <Bond atomName1="CB" atomName2="HB3"/>
      <Bond atomName1="CB" atomName2="SG"/>
      <Bond atomName1="C" atomName2="O"/>
      <ExternalBond atomName="SG"/>
      <ExternalBond atomName="C"/>
    </Residue>
    <Residue name="NGLN">
      <Atom charge="0.1493" name="N" type="N287"/>
      <Atom charge="0.1996" name="H1" type="H290"/>
      <Atom charge="0.1996" name="H2" type="H290"/>
      <Atom charge="0.1996" name="H3" type="H290"/>
      <Atom charge="0.0536" name="CA" type="C293"/>
      <Atom charge="0.1015" name="HA" type="H140"/>
      <Atom charge="0.0651" name="CB" type="C136"/>
      <Atom charge="0.005" name="HB2" type="H140"/>
      <Atom charge="0.005" name="HB3" type="H140"/>
      <Atom charge="-0.0903" name="CG" type="C136"/>
      <Atom charge="0.0331" name="HG2" type="H140"/>
      <Atom charge="0.0331" name="HG3" type="H140"/>
      <Atom charge="0.7354" name="CD" type="C235"/>
      <Atom charge="-0.6133" name="OE1" type="O236"/>
      <Atom charge="-1.0031" name="NE2" type="N237"/>
      <Atom charge="0.4429" name="HE21" type="H240"/>
      <Atom charge="0.4429" name="HE22" type="H240"/>
      <Atom charge="0.6123" name="C" type="C235"/>
      <Atom charge="-0.5713" name="O" type="O236"/>
      <Bond atomName1="N" atomName2="H1"/>
      <Bond atomName1="N" atomName2="H2"/>
      <Bond atomName1="N" atomName2="H3"/>
      <Bond atomName1="N" atomName2="CA"/>
      <Bond atomName1="CA" atomName2="HA"/>
      <Bond atomName1="CA" atomName2="CB"/>
      <Bond atomName1="CA" atomName2="C"/>
      <Bond atomName1="CB" atomName2="HB2"/>
      <Bond atomName1="CB" atomName2="HB3"/>
      <Bond atomName1="CB" atomName2="CG"/>
      <Bond atomName1="CG" atomName2="HG2"/>
      <Bond atomName1="CG" atomName2="HG3"/>
      <Bond atomName1="CG" atomName2="CD"/>
      <Bond atomName1="CD" atomName2="OE1"/>
      <Bond atomName1="CD" atomName2="NE2"/>
      <Bond atomName1="NE2" atomName2="HE21"/>
      <Bond atomName1="NE2" atomName2="HE22"/>
      <Bond atomName1="C" atomName2="O"/>
      <ExternalBond atomName="C"/>
    </Residue>
    <Residue name="NGLU">
      <Atom charge="0.0017" name="N" type="N287"/>
      <Atom charge="0.2391" name="H1" type="H290"/>
      <Atom charge="0.2391" name="H2" type="H290"/>
      <Atom charge="0.2391" name="H3" type="H290"/>
      <Atom charge="0.0588" name="CA" type="C293"/>
      <Atom charge="0.1202" name="HA" type="H140"/>
      <Atom charge="0.0909" name="CB" type="C136"/>
      <Atom charge="-0.0232" name="HB2" type="H140"/>
      <Atom charge="-0.0232" name="HB3" type="H140"/>
      <Atom charge="-0.0236" name="CG" type="C274"/>
      <Atom charge="-0.0315" name="HG2" type="H140"/>
      <Atom charge="-0.0315" name="HG3" type="H140"/>
      <Atom charge="0.8087" name="CD" type="C271"/>
      <Atom charge="-0.8189" name="OE1" type="O272"/>
      <Atom charge="-0.8189" name="OE2" type="O272"/>
      <Atom charge="0.5621" name="C" type="C235"/>
      <Atom charge="-0.5889" name="O" type="O236"/>
      <Bond atomName1="N" atomName2="H1"/>
      <Bond atomName1="N" atomName2="H2"/>
      <Bond atomName1="N" atomName2="H3"/>
      <Bond atomName1="N" atomName2="CA"/>
      <Bond atomName1="CA" atomName2="HA"/>
      <Bond atomName1="CA" atomName2="CB"/>
      <Bond atomName1="CA" atomName2="C"/>
      <Bond atomName1="CB" atomName2="HB2"/>
      <Bond atomName1="CB" atomName2="HB3"/>
      <Bond atomName1="CB" atomName2="CG"/>
      <Bond atomName1="CG" atomName2="HG2"/>
      <Bond atomName1="CG" atomName2="HG3"/>
      <Bond atomName1="CG" atomName2="CD"/>
      <Bond atomName1="CD" atomName2="OE1"/>
      <Bond atomName1="CD" atomName2="OE2"/>
      <Bond atomName1="C" atomName2="O"/>
      <ExternalBond atomName="C"/>
    </Residue>
    <Residue name="NGLY">
      <Atom charge="0.2943" name="N" type="N287"/>
      <Atom charge="0.1642" name="H1" type="H290"/>
      <Atom charge="0.1642" name="H2" type="H290"/>
      <Atom charge="0.1642" name="H3" type="H290"/>
      <Atom charge="-0.01" name="CA" type="C292"/>
      <Atom charge="0.0895" name="HA2" type="H140"/>
      <Atom charge="0.0895" name="HA3" type="H140"/>
      <Atom charge="0.6163" name="C" type="C235"/>
      <Atom charge="-0.5722" name="O" type="O236"/>
      <Bond atomName1="N" atomName2="H1"/>
      <Bond atomName1="N" atomName2="H2"/>
      <Bond atomName1="N" atomName2="H3"/>
      <Bond atomName1="N" atomName2="CA"/>
      <Bond atomName1="CA" atomName2="HA2"/>
      <Bond atomName1="CA" atomName2="HA3"/>
      <Bond atomName1="CA" atomName2="C"/>
      <Bond atomName1="C" atomName2="O"/>
      <ExternalBond atomName="C"/>
    </Residue>
    <Residue name="NHID">
      <Atom charge="0.1542" name="N" type="N287"/>
      <Atom charge="0.1963" name="H1" type="H290"/>
      <Atom charge="0.1963" name="H2" type="H290"/>
      <Atom charge="0.1963" name="H3" type="H290"/>
      <Atom charge="0.0964" name="CA" type="C293"/>
      <Atom charge="0.0958" name="HA" type="H140"/>
      <Atom charge="0.0259" name="CB" type="C505"/>
      <Atom charge="0.0209" name="HB2" type="H140"/>
      <Atom charge="0.0209" name="HB3" type="H140"/>
      <Atom charge="-0.0399" name="CG" type="C508"/>
      <Atom charge="-0.3819" name="ND1" type="N503"/>
      <Atom charge="0.3632" name="HD1" type="H504"/>
      <Atom charge="0.2127" name="CE1" type="C506"/>
      <Atom charge="0.1385" name="HE1" type="H146"/>
      <Atom charge="-0.5711" name="NE2" type="N511"/>
      <Atom charge="0.1046" name="CD2" type="C507"/>
      <Atom charge="0.1299" name="HD2" type="H146"/>
      <Atom charge="0.6123" name="C" type="C235"/>
      <Atom charge="-0.5713" name="O" type="O236"/>
      <Bond atomName1="N" atomName2="H1"/>
      <Bond atomName1="N" atomName2="H2"/>
      <Bond atomName1="N" atomName2="H3"/>
      <Bond atomName1="N" atomName2="CA"/>
      <Bond atomName1="CA" atomName2="HA"/>
      <Bond atomName1="CA" atomName2="CB"/>
      <Bond atomName1="CA" atomName2="C"/>
      <Bond atomName1="CB" atomName2="HB2"/>
      <Bond atomName1="CB" atomName2="HB3"/>
      <Bond atomName1="CB" atomName2="CG"/>
      <Bond atomName1="CG" atomName2="ND1"/>
      <Bond atomName1="CG" atomName2="CD2"/>
      <Bond atomName1="ND1" atomName2="HD1"/>
      <Bond atomName1="ND1" atomName2="CE1"/>
      <Bond atomName1="CE1" atomName2="HE1"/>
      <Bond atomName1="CE1" atomName2="NE2"/>
      <Bond atomName1="NE2" atomName2="CD2"/>
      <Bond atomName1="CD2" atomName2="HD2"/>
      <Bond atomName1="C" atomName2="O"/>
      <ExternalBond atomName="C"/>
    </Residue>
    <Residue name="NHIE">
      <Atom charge="0.1472" name="N" type="N287"/>
      <Atom charge="0.2016" name="H1" type="H290"/>
      <Atom charge="0.2016" name="H2" type="H290"/>
      <Atom charge="0.2016" name="H3" type="H290"/>
      <Atom charge="0.0236" name="CA" type="C293"/>
      <Atom charge="0.138" name="HA" type="H140"/>
      <Atom charge="0.0489" name="CB" type="C505"/>
      <Atom charge="0.0223" name="HB2" type="H140"/>
      <Atom charge="0.0223" name="HB3" type="H140"/>
      <Atom charge="0.174" name="CG" type="C507"/>
      <Atom charge="-0.5579" name="ND1" type="N511"/>
      <Atom charge="0.1804" name="CE1" type="C506"/>
      <Atom charge="0.1397" name="HE1" type="H146"/>
      <Atom charge="-0.2781" name="NE2" type="N503"/>
      <Atom charge="0.3324" name="HE2" type="H504"/>
      <Atom charge="-0.2349" name="CD2" type="C508"/>
      <Atom charge="0.1963" name="HD2" type="H146"/>
      <Atom charge="0.6123" name="C" type="C235"/>
      <Atom charge="-0.5713" name="O" type="O236"/>
      <Bond atomName1="N" atomName2="H1"/>
      <Bond atomName1="N" atomName2="H2"/>
      <Bond atomName1="N" atomName2="H3"/>
      <Bond atomName1="N" atomName2="CA"/>
      <Bond atomName1="CA" atomName2="HA"/>
      <Bond atomName1="CA" atomName2="CB"/>
      <Bond atomName1="CA" atomName2="C"/>
      <Bond atomName1="CB" atomName2="HB2"/>
      <Bond atomName1="CB" atomName2="HB3"/>
      <Bond atomName1="CB" atomName2="CG"/>
      <Bond atomName1="CG" atomName2="ND1"/>
      <Bond atomName1="CG" atomName2="CD2"/>
      <Bond atomName1="ND1" atomName2="CE1"/>
      <Bond atomName1="CE1" atomName2="HE1"/>
      <Bond atomName1="CE1" atomName2="NE2"/>
      <Bond atomName1="NE2" atomName2="HE2"/>
      <Bond atomName1="NE2" atomName2="CD2"/>
      <Bond atomName1="CD2" atomName2="HD2"/>
      <Bond atomName1="C" atomName2="O"/>
      <ExternalBond atomName="C"/>
    </Residue>
    <Residue name="NHIP">
      <Atom charge="0.256" name="N" type="N287"/>
      <Atom charge="0.1704" name="H1" type="H290"/>
      <Atom charge="0.1704" name="H2" type="H290"/>
      <Atom charge="0.1704" name="H3" type="H290"/>
      <Atom charge="0.0581" name="CA" type="C293"/>
      <Atom charge="0.1047" name="HA" type="H140"/>
      <Atom charge="0.0484" name="CB" type="C505"/>
      <Atom charge="0.0531" name="HB2" type="H140"/>
      <Atom charge="0.0531" name="HB3" type="H140"/>
      <Atom charge="-0.0236" name="CG" type="C510"/>
      <Atom charge="-0.151" name="ND1" type="N512"/>
      <Atom charge="0.3821" name="HD1" type="H513"/>
      <Atom charge="-0.0011" name="CE1" type="C509"/>
      <Atom charge="0.2645" name="HE1" type="H146"/>
      <Atom charge="-0.1739" name="NE2" type="N512"/>
      <Atom charge="0.3921" name="HE2" type="H513"/>
      <Atom charge="-0.1433" name="CD2" type="C510"/>
      <Atom charge="0.2495" name="HD2" type="H146"/>
      <Atom charge="0.7214" name="C" type="C235"/>
      <Atom charge="-0.6013" name="O" type="O236"/>
      <Bond atomName1="N" atomName2="H1"/>
      <Bond atomName1="N" atomName2="H2"/>
      <Bond atomName1="N" atomName2="H3"/>
      <Bond atomName1="N" atomName2="CA"/>
      <Bond atomName1="CA" atomName2="HA"/>
      <Bond atomName1="CA" atomName2="CB"/>
      <Bond atomName1="CA" atomName2="C"/>
      <Bond atomName1="CB" atomName2="HB2"/>
      <Bond atomName1="CB" atomName2="HB3"/>
      <Bond atomName1="CB" atomName2="CG"/>
      <Bond atomName1="CG" atomName2="ND1"/>
      <Bond atomName1="CG" atomName2="CD2"/>
      <Bond atomName1="ND1" atomName2="HD1"/>
      <Bond atomName1="ND1" atomName2="CE1"/>
      <Bond atomName1="CE1" atomName2="HE1"/>
      <Bond atomName1="CE1" atomName2="NE2"/>
      <Bond atomName1="NE2" atomName2="HE2"/>
      <Bond atomName1="NE2" atomName2="CD2"/>
      <Bond atomName1="CD2" atomName2="HD2"/>
      <Bond atomName1="C" atomName2="O"/>
      <ExternalBond atomName="C"/>
    </Residue>
    <Residue name="NILE">
      <Atom charge="0.0311" name="N" type="N287"/>
      <Atom charge="0.2329" name="H1" type="H290"/>
      <Atom charge="0.2329" name="H2" type="H290"/>
      <Atom charge="0.2329" name="H3" type="H290"/>
      <Atom charge="0.0257" name="CA" type="C293"/>
      <Atom charge="0.1031" name="HA" type="H140"/>
      <Atom charge="0.1885" name="CB" type="C137"/>
      <Atom charge="0.0213" name="HB" type="H140"/>
      <Atom charge="-0.372" name="CG2" type="C135"/>
      <Atom charge="0.0947" name="HG21" type="H140"/>
      <Atom charge="0.0947" name="HG22" type="H140"/>
      <Atom charge="0.0947" name="HG23" type="H140"/>
      <Atom charge="-0.0387" name="CG1" type="C136"/>
      <Atom charge="0.0201" name="HG12" type="H140"/>
      <Atom charge="0.0201" name="HG13" type="H140"/>
      <Atom charge="-0.0908" name="CD1" type="C135"/>
      <Atom charge="0.0226" name="HD11" type="H140"/>
      <Atom charge="0.0226" name="HD12" type="H140"/>
      <Atom charge="0.0226" name="HD13" type="H140"/>
      <Atom charge="0.6123" name="C" type="C235"/>
      <Atom charge="-0.5713" name="O" type="O236"/>
      <Bond atomName1="N" atomName2="H1"/>
      <Bond atomName1="N" atomName2="H2"/>
      <Bond atomName1="N" atomName2="H3"/>
      <Bond atomName1="N" atomName2="CA"/>
      <Bond atomName1="CA" atomName2="HA"/>
      <Bond atomName1="CA" atomName2="CB"/>
      <Bond atomName1="CA" atomName2="C"/>
      <Bond atomName1="CB" atomName2="HB"/>
      <Bond atomName1="CB" atomName2="CG2"/>
      <Bond atomName1="CB" atomName2="CG1"/>
      <Bond atomName1="CG2" atomName2="HG21"/>
      <Bond atomName1="CG2" atomName2="HG22"/>
      <Bond atomName1="CG2" atomName2="HG23"/>
      <Bond atomName1="CG1" atomName2="HG12"/>
      <Bond atomName1="CG1" atomName2="HG13"/>
      <Bond atomName1="CG1" atomName2="CD1"/>
      <Bond atomName1="CD1" atomName2="HD11"/>
      <Bond atomName1="CD1" atomName2="HD12"/>
      <Bond atomName1="CD1" atomName2="HD13"/>
      <Bond atomName1="C" atomName2="O"/>
      <ExternalBond atomName="C"/>
    </Residue>
    <Residue name="NLEU">
      <Atom charge="0.101" name="N" type="N287"/>
      <Atom charge="0.2148" name="H1" type="H290"/>
      <Atom charge="0.2148" name="H2" type="H290"/>
      <Atom charge="0.2148" name="H3" type="H290"/>
      <Atom charge="0.0104" name="CA" type="C293"/>
      <Atom charge="0.1053" name="HA" type="H140"/>
      <Atom charge="-0.0244" name="CB" type="C136"/>
      <Atom charge="0.0256" name="HB2" type="H140"/>
      <Atom charge="0.0256" name="HB3" type="H140"/>
      <Atom charge="0.3421" name="CG" type="C137"/>
      <Atom charge="-0.038" name="HG" type="H140"/>
      <Atom charge="-0.4106" name="CD1" type="C135"/>
      <Atom charge="0.098" name="HD11" type="H140"/>
      <Atom charge="0.098" name="HD12" type="H140"/>
      <Atom charge="0.098" name="HD13" type="H140"/>
      <Atom charge="-0.4104" name="CD2" type="C135"/>
      <Atom charge="0.098" name="HD21" type="H140"/>
      <Atom charge="0.098" name="HD22" type="H140"/>
      <Atom charge="0.098" name="HD23" type="H140"/>
      <Atom charge="0.6123" name="C" type="C235"/>
      <Atom charge="-0.5713" name="O" type="O236"/>
      <Bond atomName1="N" atomName2="H1"/>
      <Bond atomName1="N" atomName2="H2"/>
      <Bond atomName1="N" atomName2="H3"/>
      <Bond atomName1="N" atomName2="CA"/>
      <Bond atomName1="CA" atomName2="HA"/>
      <Bond atomName1="CA" atomName2="CB"/>
      <Bond atomName1="CA" atomName2="C"/>
      <Bond atomName1="CB" atomName2="HB2"/>
      <Bond atomName1="CB" atomName2="HB3"/>
      <Bond atomName1="CB" atomName2="CG"/>
      <Bond atomName1="CG" atomName2="HG"/>
      <Bond atomName1="CG" atomName2="CD1"/>
      <Bond atomName1="CG" atomName2="CD2"/>
      <Bond atomName1="CD1" atomName2="HD11"/>
      <Bond atomName1="CD1" atomName2="HD12"/>
      <Bond atomName1="CD1" atomName2="HD13"/>
      <Bond atomName1="CD2" atomName2="HD21"/>
      <Bond atomName1="CD2" atomName2="HD22"/>
      <Bond atomName1="CD2" atomName2="HD23"/>
      <Bond atomName1="C" atomName2="O"/>
      <ExternalBond atomName="C"/>
    </Residue>
    <Residue name="NLYS">
      <Atom charge="0.0966" name="N" type="N287"/>
      <Atom charge="0.2165" name="H1" type="H290"/>
      <Atom charge="0.2165" name="H2" type="H290"/>
      <Atom charge="0.2165" name="H3" type="H290"/>
      <Atom charge="-0.0015" name="CA" type="C293"/>
      <Atom charge="0.118" name="HA" type="H140"/>
      <Atom charge="0.0212" name="CB" type="C136"/>
      <Atom charge="0.0283" name="HB2" type="H140"/>
      <Atom charge="0.0283" name="HB3" type="H140"/>
      <Atom charge="-0.0048" name="CG" type="C136"/>
      <Atom charge="0.0121" name="HG2" type="H140"/>
      <Atom charge="0.0121" name="HG3" type="H140"/>
      <Atom charge="-0.0608" name="CD" type="C136"/>
      <Atom charge="0.0633" name="HD2" type="H140"/>
      <Atom charge="0.0633" name="HD3" type="H140"/>
      <Atom charge="-0.0181" name="CE" type="C296"/>
      <Atom charge="0.1171" name="HE2" type="H140"/>
      <Atom charge="0.1171" name="HE3" type="H140"/>
      <Atom charge="-0.3764" name="NZ" type="N287"/>
      <Atom charge="0.3382" name="HZ1" type="H290"/>
      <Atom charge="0.3382" name="HZ2" type="H290"/>
      <Atom charge="0.3382" name="HZ3" type="H290"/>
      <Atom charge="0.7214" name="C" type="C235"/>
      <Atom charge="-0.6013" name="O" type="O236"/>
      <Bond atomName1="N" atomName2="H1"/>
      <Bond atomName1="N" atomName2="H2"/>
      <Bond atomName1="N" atomName2="H3"/>
      <Bond atomName1="N" atomName2="CA"/>
      <Bond atomName1="CA" atomName2="HA"/>
      <Bond atomName1="CA" atomName2="CB"/>
      <Bond atomName1="CA" atomName2="C"/>
      <Bond atomName1="CB" atomName2="HB2"/>
      <Bond atomName1="CB" atomName2="HB3"/>
      <Bond atomName1="CB" atomName2="CG"/>
      <Bond atomName1="CG" atomName2="HG2"/>
      <Bond atomName1="CG" atomName2="HG3"/>
      <Bond atomName1="CG" atomName2="CD"/>
      <Bond atomName1="CD" atomName2="HD2"/>
      <Bond atomName1="CD" atomName2="HD3"/>
      <Bond atomName1="CD" atomName2="CE"/>
      <Bond atomName1="CE" atomName2="HE2"/>
      <Bond atomName1="CE" atomName2="HE3"/>
      <Bond atomName1="CE" atomName2="NZ"/>
      <Bond atomName1="NZ" atomName2="HZ1"/>
      <Bond atomName1="NZ" atomName2="HZ2"/>
      <Bond atomName1="NZ" atomName2="HZ3"/>
      <Bond atomName1="C" atomName2="O"/>
      <ExternalBond atomName="C"/>
    </Residue>
    <Residue name="NMET">
      <Atom charge="0.1592" name="N" type="N287"/>
      <Atom charge="0.1984" name="H1" type="H290"/>
      <Atom charge="0.1984" name="H2" type="H290"/>
      <Atom charge="0.1984" name="H3" type="H290"/>
      <Atom charge="0.0221" name="CA" type="C293"/>
      <Atom charge="0.1116" name="HA" type="H140"/>
      <Atom charge="0.0865" name="CB" type="C136"/>
      <Atom charge="0.0125" name="HB2" type="H140"/>
      <Atom charge="0.0125" name="HB3" type="H140"/>
      <Atom charge="0.0334" name="CG" type="C210"/>
      <Atom charge="0.0292" name="HG2" type="H140"/>
      <Atom charge="0.0292" name="HG3" type="H140"/>
      <Atom charge="-0.2774" name="SD" type="S202"/>
      <Atom charge="-0.0341" name="CE" type="C209"/>
      <Atom charge="0.0597" name="HE1" type="H140"/>
      <Atom charge="0.0597" name="HE2" type="H140"/>
      <Atom charge="0.0597" name="HE3" type="H140"/>
      <Atom charge="0.6123" name="C" type="C235"/>
      <Atom charge="-0.5713" name="O" type="O236"/>
      <Bond atomName1="N" atomName2="H1"/>
      <Bond atomName1="N" atomName2="H2"/>
      <Bond atomName1="N" atomName2="H3"/>
      <Bond atomName1="N" atomName2="CA"/>
      <Bond atomName1="CA" atomName2="HA"/>
      <Bond atomName1="CA" atomName2="CB"/>
      <Bond atomName1="CA" atomName2="C"/>
      <Bond atomName1="CB" atomName2="HB2"/>
      <Bond atomName1="CB" atomName2="HB3"/>
      <Bond atomName1="CB" atomName2="CG"/>
      <Bond atomName1="CG" atomName2="HG2"/>
      <Bond atomName1="CG" atomName2="HG3"/>
      <Bond atomName1="CG" atomName2="SD"/>
      <Bond atomName1="SD" atomName2="CE"/>
      <Bond atomName1="CE" atomName2="HE1"/>
      <Bond atomName1="CE" atomName2="HE2"/>
      <Bond atomName1="CE" atomName2="HE3"/>
      <Bond atomName1="C" atomName2="O"/>
      <ExternalBond atomName="C"/>
    </Residue>
    <Residue name="NPHE">
      <Atom charge="0.1737" name="N" type="N287"/>
      <Atom charge="0.1921" name="H1" type="H290"/>
      <Atom charge="0.1921" name="H2" type="H290"/>
      <Atom charge="0.1921" name="H3" type="H290"/>
      <Atom charge="0.0733" name="CA" type="C293"/>
      <Atom charge="0.1041" name="HA" type="H140"/>
      <Atom charge="0.033" name="CB" type="C149"/>
      <Atom charge="0.0104" name="HB2" type="H140"/>
      <Atom charge="0.0104" name="HB3" type="H140"/>
      <Atom charge="0.0031" name="CG" type="C145"/>
      <Atom charge="-0.1392" name="CD1" type="C145"/>
      <Atom charge="0.1374" name="HD1" type="H146"/>
      <Atom charge="-0.1602" name="CE1" type="C145"/>
      <Atom charge="0.1433" name="HE1" type="H146"/>
      <Atom charge="-0.1208" name="CZ" type="C145"/>
      <Atom charge="0.1329" name="HZ" type="H146"/>
      <Atom charge="-0.1603" name="CE2" type="C145"/>
      <Atom charge="0.1433" name="HE2" type="H146"/>
      <Atom charge="-0.1391" name="CD2" type="C145"/>
      <Atom charge="0.1374" name="HD2" type="H146"/>
      <Atom charge="0.6123" name="C" type="C235"/>
      <Atom charge="-0.5713" name="O" type="O236"/>
      <Bond atomName1="N" atomName2="H1"/>
      <Bond atomName1="N" atomName2="H2"/>
      <Bond atomName1="N" atomName2="H3"/>
      <Bond atomName1="N" atomName2="CA"/>
      <Bond atomName1="CA" atomName2="HA"/>
      <Bond atomName1="CA" atomName2="CB"/>
      <Bond atomName1="CA" atomName2="C"/>
      <Bond atomName1="CB" atomName2="HB2"/>
      <Bond atomName1="CB" atomName2="HB3"/>
      <Bond atomName1="CB" atomName2="CG"/>
      <Bond atomName1="CG" atomName2="CD1"/>
      <Bond atomName1="CG" atomName2="CD2"/>
      <Bond atomName1="CD1" atomName2="HD1"/>
      <Bond atomName1="CD1" atomName2="CE1"/>
      <Bond atomName1="CE1" atomName2="HE1"/>
      <Bond atomName1="CE1" atomName2="CZ"/>
      <Bond atomName1="CZ" atomName2="HZ"/>
      <Bond atomName1="CZ" atomName2="CE2"/>
      <Bond atomName1="CE2" atomName2="HE2"/>
      <Bond atomName1="CE2" atomName2="CD2"/>
      <Bond atomName1="CD2" atomName2="HD2"/>
      <Bond atomName1="C" atomName2="O"/>
      <ExternalBond atomName="C"/>
    </Residue>
    <Residue name="NPRO">
      <Atom charge="-0.202" name="N" type="N309"/>
      <Atom charge="0.312" name="H2" type="H310"/>
      <Atom charge="0.312" name="H3" type="H310"/>
      <Atom charge="-0.012" name="CD" type="C296"/>
      <Atom charge="0.1" name="HD2" type="H140"/>
      <Atom charge="0.1" name="HD3" type="H140"/>
      <Atom charge="-0.121" name="CG" type="C136"/>
      <Atom charge="0.1" name="HG2" type="H140"/>
      <Atom charge="0.1" name="HG3" type="H140"/>
      <Atom charge="-0.115" name="CB" type="C136"/>
      <Atom charge="0.1" name="HB2" type="H140"/>
      <Atom charge="0.1" name="HB3" type="H140"/>
      <Atom charge="0.1" name="CA" type="C295"/>
      <Atom charge="0.1" name="HA" type="H140"/>
      <Atom charge="0.526" name="C" type="C235"/>
      <Atom charge="-0.5" name="O" type="O236"/>
      <Bond atomName1="N" atomName2="H2"/>
      <Bond atomName1="N" atomName2="H3"/>
      <Bond atomName1="N" atomName2="CD"/>
      <Bond atomName1="N" atomName2="CA"/>
      <Bond atomName1="CD" atomName2="HD2"/>
      <Bond atomName1="CD" atomName2="HD3"/>
      <Bond atomName1="CD" atomName2="CG"/>
      <Bond atomName1="CG" atomName2="HG2"/>
      <Bond atomName1="CG" atomName2="HG3"/>
      <Bond atomName1="CG" atomName2="CB"/>
      <Bond atomName1="CB" atomName2="HB2"/>
      <Bond atomName1="CB" atomName2="HB3"/>
      <Bond atomName1="CB" atomName2="CA"/>
      <Bond atomName1="CA" atomName2="HA"/>
      <Bond atomName1="CA" atomName2="C"/>
      <Bond atomName1="C" atomName2="O"/>
      <ExternalBond atomName="C"/>
    </Residue>
    <Residue name="NSER">
      <Atom charge="0.1849" name="N" type="N287"/>
      <Atom charge="0.1898" name="H1" type="H290"/>
      <Atom charge="0.1898" name="H2" type="H290"/>
      <Atom charge="0.1898" name="H3" type="H290"/>
      <Atom charge="0.0567" name="CA" type="C293"/>
      <Atom charge="0.0782" name="HA" type="H140"/>
      <Atom charge="0.2596" name="CB" type="C157"/>
      <Atom charge="0.0273" name="HB2" type="H140"/>
      <Atom charge="0.0273" name="HB3" type="H140"/>
      <Atom charge="-0.6714" name="OG" type="O154"/>
      <Atom charge="0.4239" name="HG" type="H155"/>
      <Atom charge="0.6163" name="C" type="C235"/>
      <Atom charge="-0.5722" name="O" type="O236"/>
      <Bond atomName1="N" atomName2="H1"/>
      <Bond atomName1="N" atomName2="H2"/>
      <Bond atomName1="N" atomName2="H3"/>
      <Bond atomName1="N" atomName2="CA"/>
      <Bond atomName1="CA" atomName2="HA"/>
      <Bond atomName1="CA" atomName2="CB"/>
      <Bond atomName1="CA" atomName2="C"/>
      <Bond atomName1="CB" atomName2="HB2"/>
      <Bond atomName1="CB" atomName2="HB3"/>
      <Bond atomName1="CB" atomName2="OG"/>
      <Bond atomName1="OG" atomName2="HG"/>
      <Bond atomName1="C" atomName2="O"/>
      <ExternalBond atomName="C"/>
    </Residue>
    <Residue name="NTHR">
      <Atom charge="0.1812" name="N" type="N287"/>
      <Atom charge="0.1934" name="H1" type="H290"/>
      <Atom charge="0.1934" name="H2" type="H290"/>
      <Atom charge="0.1934" name="H3" type="H290"/>
      <Atom charge="0.0034" name="CA" type="C293"/>
      <Atom charge="0.1087" name="HA" type="H140"/>
      <Atom charge="0.4514" name="CB" type="C158"/>
      <Atom charge="-0.0323" name="HB" type="H140"/>
      <Atom charge="-0.2554" name="CG2" type="C135"/>
      <Atom charge="0.0627" name="HG21" type="H140"/>
      <Atom charge="0.0627" name="HG22" type="H140"/>
      <Atom charge="0.0627" name="HG23" type="H140"/>
      <Atom charge="-0.6764" name="OG1" type="O154"/>
      <Atom charge="0.407" name="HG1" type="H155"/>
      <Atom charge="0.6163" name="C" type="C235"/>
      <Atom charge="-0.5722" name="O" type="O236"/>
      <Bond atomName1="N" atomName2="H1"/>
      <Bond atomName1="N" atomName2="H2"/>
      <Bond atomName1="N" atomName2="H3"/>
      <Bond atomName1="N" atomName2="CA"/>
      <Bond atomName1="CA" atomName2="HA"/>
      <Bond atomName1="CA" atomName2="CB"/>
      <Bond atomName1="CA" atomName2="C"/>
      <Bond atomName1="CB" atomName2="HB"/>
      <Bond atomName1="CB" atomName2="CG2"/>
      <Bond atomName1="CB" atomName2="OG1"/>
      <Bond atomName1="CG2" atomName2="HG21"/>
      <Bond atomName1="CG2" atomName2="HG22"/>
      <Bond atomName1="CG2" atomName2="HG23"/>
      <Bond atomName1="OG1" atomName2="HG1"/>
      <Bond atomName1="C" atomName2="O"/>
      <ExternalBond atomName="C"/>
    </Residue>
    <Residue name="NTRP">
      <Atom charge="0.1913" name="N" type="N287"/>
      <Atom charge="0.1888" name="H1" type="H290"/>
      <Atom charge="0.1888" name="H2" type="H290"/>
      <Atom charge="0.1888" name="H3" type="H290"/>
      <Atom charge="0.0421" name="CA" type="C293"/>
      <Atom charge="0.1162" name="HA" type="H140"/>
      <Atom charge="0.0543" name="CB" type="C136"/>
      <Atom charge="0.0222" name="HB2" type="H140"/>
      <Atom charge="0.0222" name="HB3" type="H140"/>
      <Atom charge="-0.1654" name="CG" type="C500"/>
      <Atom charge="-0.1788" name="CD1" type="C514"/>
      <Atom charge="0.2195" name="HD1" type="H146"/>
      <Atom charge="-0.3444" name="NE1" type="N503"/>
      <Atom charge="0.3412" name="HE1" type="H504"/>
      <Atom charge="0.1575" name="CE2" type="C502"/>
      <Atom charge="-0.271" name="CZ2" type="C145"/>
      <Atom charge="0.1589" name="HZ2" type="H146"/>
      <Atom charge="-0.108" name="CH2" type="C145"/>
      <Atom charge="0.1411" name="HH2" type="H146"/>
      <Atom charge="-0.2034" name="CZ3" type="C145"/>
      <Atom charge="0.1458" name="HZ3" type="H146"/>
      <Atom charge="-0.2265" name="CE3" type="C145"/>
      <Atom charge="0.1646" name="HE3" type="H146"/>
      <Atom charge="0.1132" name="CD2" type="C501"/>
      <Atom charge="0.6123" name="C" type="C235"/>
      <Atom charge="-0.5713" name="O" type="O236"/>
      <Bond atomName1="N" atomName2="H1"/>
      <Bond atomName1="N" atomName2="H2"/>
      <Bond atomName1="N" atomName2="H3"/>
      <Bond atomName1="N" atomName2="CA"/>
      <Bond atomName1="CA" atomName2="HA"/>
      <Bond atomName1="CA" atomName2="CB"/>
      <Bond atomName1="CA" atomName2="C"/>
      <Bond atomName1="CB" atomName2="HB2"/>
      <Bond atomName1="CB" atomName2="HB3"/>
      <Bond atomName1="CB" atomName2="CG"/>
      <Bond atomName1="CG" atomName2="CD1"/>
      <Bond atomName1="CG" atomName2="CD2"/>
      <Bond atomName1="CD1" atomName2="HD1"/>
      <Bond atomName1="CD1" atomName2="NE1"/>
      <Bond atomName1="NE1" atomName2="HE1"/>
      <Bond atomName1="NE1" atomName2="CE2"/>
      <Bond atomName1="CE2" atomName2="CZ2"/>
      <Bond atomName1="CE2" atomName2="CD2"/>
      <Bond atomName1="CZ2" atomName2="HZ2"/>
      <Bond atomName1="CZ2" atomName2="CH2"/>
      <Bond atomName1="CH2" atomName2="HH2"/>
      <Bond atomName1="CH2" atomName2="CZ3"/>
      <Bond atomName1="CZ3" atomName2="HZ3"/>
      <Bond atomName1="CZ3" atomName2="CE3"/>
      <Bond atomName1="CE3" atomName2="HE3"/>
      <Bond atomName1="CE3" atomName2="CD2"/>
      <Bond atomName1="C" atomName2="O"/>
      <ExternalBond atomName="C"/>
    </Residue>
    <Residue name="NTYR">
      <Atom charge="0.194" name="N" type="N287"/>
      <Atom charge="0.1873" name="H1" type="H290"/>
      <Atom charge="0.1873" name="H2" type="H290"/>
      <Atom charge="0.1873" name="H3" type="H290"/>
      <Atom charge="0.057" name="CA" type="C293"/>
      <Atom charge="0.0983" name="HA" type="H140"/>
      <Atom charge="0.0659" name="CB" type="C149"/>
      <Atom charge="0.0102" name="HB2" type="H140"/>
      <Atom charge="0.0102" name="HB3" type="H140"/>
      <Atom charge="-0.0205" name="CG" type="C145"/>
      <Atom charge="-0.2002" name="CD1" type="C145"/>
      <Atom charge="0.172" name="HD1" type="H146"/>
      <Atom charge="-0.2239" name="CE1" type="C145"/>
      <Atom charge="0.165" name="HE1" type="H146"/>
      <Atom charge="0.3139" name="CZ" type="C166"/>
      <Atom charge="-0.5578" name="OH" type="O167"/>
      <Atom charge="0.4001" name="HH" type="H168"/>
      <Atom charge="-0.2239" name="CE2" type="C145"/>
      <Atom charge="0.165" name="HE2" type="H146"/>
      <Atom charge="-0.2002" name="CD2" type="C145"/>
      <Atom charge="0.172" name="HD2" type="H146"/>
      <Atom charge="0.6123" name="C" type="C235"/>
      <Atom charge="-0.5713" name="O" type="O236"/>
      <Bond atomName1="N" atomName2="H1"/>
      <Bond atomName1="N" atomName2="H2"/>
      <Bond atomName1="N" atomName2="H3"/>
      <Bond atomName1="N" atomName2="CA"/>
      <Bond atomName1="CA" atomName2="HA"/>
      <Bond atomName1="CA" atomName2="CB"/>
      <Bond atomName1="CA" atomName2="C"/>
      <Bond atomName1="CB" atomName2="HB2"/>
      <Bond atomName1="CB" atomName2="HB3"/>
      <Bond atomName1="CB" atomName2="CG"/>
      <Bond atomName1="CG" atomName2="CD1"/>
      <Bond atomName1="CG" atomName2="CD2"/>
      <Bond atomName1="CD1" atomName2="HD1"/>
      <Bond atomName1="CD1" atomName2="CE1"/>
      <Bond atomName1="CE1" atomName2="HE1"/>
      <Bond atomName1="CE1" atomName2="CZ"/>
      <Bond atomName1="CZ" atomName2="OH"/>
      <Bond atomName1="CZ" atomName2="CE2"/>
      <Bond atomName1="OH" atomName2="HH"/>
      <Bond atomName1="CE2" atomName2="HE2"/>
      <Bond atomName1="CE2" atomName2="CD2"/>
      <Bond atomName1="CD2" atomName2="HD2"/>
      <Bond atomName1="C" atomName2="O"/>
      <ExternalBond atomName="C"/>
    </Residue>
    <Residue name="NVAL">
      <Atom charge="0.0577" name="N" type="N287"/>
      <Atom charge="0.2272" name="H1" type="H290"/>
      <Atom charge="0.2272" name="H2" type="H290"/>
      <Atom charge="0.2272" name="H3" type="H290"/>
      <Atom charge="-0.0054" name="CA" type="C293"/>
      <Atom charge="0.1093" name="HA" type="H140"/>
      <Atom charge="0.3196" name="CB" type="C137"/>
      <Atom charge="-0.0221" name="HB" type="H140"/>
      <Atom charge="-0.3129" name="CG1" type="C135"/>
      <Atom charge="0.0735" name="HG11" type="H140"/>
      <Atom charge="0.0735" name="HG12" type="H140"/>
      <Atom charge="0.0735" name="HG13" type="H140"/>
      <Atom charge="-0.3129" name="CG2" type="C135"/>
      <Atom charge="0.0735" name="HG21" type="H140"/>
      <Atom charge="0.0735" name="HG22" type="H140"/>
      <Atom charge="0.0735" name="HG23" type="H140"/>
      <Atom charge="0.6163" name="C" type="C235"/>
      <Atom charge="-0.5722" name="O" type="O236"/>
      <Bond atomName1="N" atomName2="H1"/>
      <Bond atomName1="N" atomName2="H2"/>
      <Bond atomName1="N" atomName2="H3"/>
      <Bond atomName1="N" atomName2="CA"/>
      <Bond atomName1="CA" atomName2="HA"/>
      <Bond atomName1="CA" atomName2="CB"/>
      <Bond atomName1="CA" atomName2="C"/>
      <Bond atomName1="CB" atomName2="HB"/>
      <Bond atomName1="CB" atomName2="CG1"/>
      <Bond atomName1="CB" atomName2="CG2"/>
      <Bond atomName1="CG1" atomName2="HG11"/>
      <Bond atomName1="CG1" atomName2="HG12"/>
      <Bond atomName1="CG1" atomName2="HG13"/>
      <Bond atomName1="CG2" atomName2="HG21"/>
      <Bond atomName1="CG2" atomName2="HG22"/>
      <Bond atomName1="CG2" atomName2="HG23"/>
      <Bond atomName1="C" atomName2="O"/>
      <ExternalBond atomName="C"/>
    </Residue>
  </Residues>
  <HarmonicBondForce>
    <Bond class1="HC" class2="CT" length="0.1092" k="273432.768"/>
    <Bond class1="CT" class2="C" length="0.1523" k="166657.088"/>
    <Bond class1="C" class2="N" length="0.1353" k="312461.12"/>
    <Bond class1="C" class2="O" length="0.122" k="581350.064"/>
    <Bond class1="N" class2="CT" length="0.1449" k="211040.96"/>
    <Bond class1="N" class2="H" length="0.1011" k="387907.008"/>
    <Bond class1="CT" class2="CT" length="0.1537" k="159109.152"/>
    <Bond class1="CT" class2="N2" length="0.1468" k="194689.888"/>
    <Bond class1="N2" class2="CA" length="0.1332" k="365104.208"/>
    <Bond class1="N2" class2="H3" length="0.1006" k="404802"/>
    <Bond class1="C" class2="OH" length="0.1357" k="228329.248"/>
    <Bond class1="OH" class2="HO" length="0.096" k="456943.008"/>
    <Bond class1="CT" class2="SH" length="0.1826" k="117963.696"/>
    <Bond class1="SH" class2="HS" length="0.1345" k="244010.88"/>
    <Bond class1="C" class2="O2" length="0.1253" k="457654.288"/>
    <Bond class1="CT" class2="CX" length="0.1492" k="204873.744"/>
    <Bond class1="CX" class2="NA" length="0.1383" k="245224.24"/>
    <Bond class1="CX" class2="CX" length="0.136" k="360024.832"/>
    <Bond class1="NA" class2="CR" length="0.1341" k="331213.808"/>
    <Bond class1="NA" class2="H" length="0.1006" k="405806.16"/>
    <Bond class1="CR" class2="HA" length="0.1078" k="312661.952"/>
    <Bond class1="CX" class2="HA" length="0.1083" k="308327.328"/>
    <Bond class1="CT" class2="CV" length="0.1494" k="210028.432"/>
    <Bond class1="CV" class2="NB" length="0.1379" k="233659.664"/>
    <Bond class1="CV" class2="CW" length="0.1364" k="349665.248"/>
    <Bond class1="NB" class2="CR" length="0.131" k="371087.328"/>
    <Bond class1="NA" class2="CW" length="0.1375" k="273800.96"/>
    <Bond class1="CW" class2="HA" length="0.1078" k="311189.184"/>
    <Bond class1="CT" class2="N3" length="0.1517" k="134791.744"/>
    <Bond class1="N3" class2="H3" length="0.1022" k="358259.184"/>
    <Bond class1="CT" class2="S" length="0.1817" k="128122.448"/>
    <Bond class1="CT" class2="CA" length="0.1505" k="189635.616"/>
    <Bond class1="CA" class2="CA" length="0.1391" k="303256.32"/>
    <Bond class1="CA" class2="HA" length="0.1085" k="291942.784"/>
    <Bond class1="CT" class2="OH" length="0.1418" k="208572.4"/>
    <Bond class1="CT" class2="CS" length="0.1495" k="204522.288"/>
    <Bond class1="CS" class2="CW" length="0.1364" k="358133.664"/>
    <Bond class1="CS" class2="CB" length="0.1439" k="223341.92"/>
    <Bond class1="NA" class2="CN" length="0.1374" k="270855.424"/>
    <Bond class1="CN" class2="CA" length="0.1396" k="292854.896"/>
    <Bond class1="CN" class2="CB" length="0.141" k="254512.72"/>
    <Bond class1="CA" class2="CB" length="0.1401" k="282947.184"/>
    <Bond class1="CA" class2="OH" length="0.136" k="277432.672"/>
    <Bond class1="S" class2="S" length="0.207" k="122415.472"/>
    <Bond class1="OT" class2="HT" length="0.09572" k="376560"/>
    <Bond class1="HT" class2="HT" length="0.15139" k="0"/>
    <!-- STILL Cv-ha this one is qubemaker USES OPLS -->
    <Bond class1="CV" class2="HA" length="0.108" k="307105.6"/>
    <!-- CT-CW  this is opls -->
    <Bond class1="CT" class2="CW" length=".1504" k="265265.6"/>
  </HarmonicBondForce>
  <HarmonicAngleForce>
    <Angle class1="HC" class2="CT" class3="C" angle="1.914434203220060" k="435.72176"/>
    <Angle class1="HC" class2="CT" class3="HC" angle="1.894973782060323" k="266.85552"/>
    <Angle class1="CT" class2="C" class3="N" angle="2.018316200298763" k="624.42016"/>
    <Angle class1="CT" class2="C" class3="O" angle="2.122302917132584" k="486.85024"/>
    <Angle class1="N" class2="C" class3="O" angle="2.143264321449037" k="506.43136"/>
    <Angle class1="C" class2="N" class3="CT" angle="2.127381825255888" k="678.47744"/>
    <Angle class1="C" class2="N" class3="H" angle="2.079891416309122" k="284.00992"/>
    <Angle class1="CT" class2="N" class3="H" angle="2.057202136033196" k="262.67152"/>
    <Angle class1="N" class2="CT" class3="C" angle="1.892006722331933" k="954.53776"/>
    <Angle class1="N" class2="CT" class3="CT" angle="1.941992952109050" k="795.96416"/>
    <Angle class1="N" class2="CT" class3="HC" angle="1.910891184838512" k="459.31952"/>
    <Angle class1="C" class2="CT" class3="CT" angle="1.955065468206488" k="961.23216"/>
    <Angle class1="CT" class2="CT" class3="HC" angle="1.902984843326977" k="378.98672"/>
    <Angle class1="CT" class2="CT" class3="CT" angle="1.963530315078660" k="847.00896"/>
    <Angle class1="CT" class2="CT" class3="N2" angle="1.911065717763711" k="1083.99072"/>
    <Angle class1="N2" class2="CT" class3="HC" angle="1.902356524796259" k="477.8128"/>
    <Angle class1="CT" class2="N2" class3="CA" angle="2.187473511402053" k="626.67952"/>
    <Angle class1="CT" class2="N2" class3="H3" angle="2.033238765403314" k="267.69232"/>
    <Angle class1="CA" class2="N2" class3="H3" angle="2.104273665959483" k="280.83008"/>
    <Angle class1="N2" class2="CA" class3="N2" angle="2.094377649100676" k="629.2736"/>
    <Angle class1="H3" class2="N2" class3="H3" angle="2.051460002794135" k="181.66928"/>
    <Angle class1="H" class2="N" class3="H" angle="2.040499335091611" k="190.70672"/>
    <Angle class1="CT" class2="C" class3="OH" angle="1.974630609121345" k="616.97264"/>
    <Angle class1="OH" class2="C" class3="O" angle="2.123594460779060" k="556.30464"/>
    <Angle class1="C" class2="OH" class3="HO" angle="1.863749841742145" k="553.37584"/>
    <Angle class1="CT" class2="CT" class3="SH" angle="2.002468610690654" k="680.23472"/>
    <Angle class1="SH" class2="CT" class3="HC" angle="1.869474521688686" k="383.2544"/>
    <Angle class1="CT" class2="SH" class3="HS" angle="1.674311804730680" k="642.82976"/>
    <Angle class1="CT" class2="C" class3="O2" angle="2.032872246260395" k="454.8008"/>
    <Angle class1="O2" class2="C" class3="O2" angle="2.217440814658796" k="468.44064"/>
    <Angle class1="CT" class2="CT" class3="CX" angle="1.991054157382611" k="820.39872"/>
    <Angle class1="CX" class2="CT" class3="HC" angle="1.907749592184922" k="401.07824"/>
    <Angle class1="CT" class2="CX" class3="NA" angle="2.126247361242092" k="603.83488"/>
    <Angle class1="CT" class2="CX" class3="CX" angle="2.312613618770046" k="525.25936"/>
    <Angle class1="NA" class2="CX" class3="CX" angle="1.854447236829015" k="542.49744"/>
    <Angle class1="CX" class2="NA" class3="CR" angle="1.924469846419027" k="545.50992"/>
    <Angle class1="CX" class2="NA" class3="H" angle="2.185571102517379" k="274.55408"/>
    <Angle class1="CR" class2="NA" class3="H" angle="2.188066923347731" k="252.12784"/>
    <Angle class1="NA" class2="CR" class3="NA" angle="1.866961247565814" k="467.60384"/>
    <Angle class1="NA" class2="CR" class3="HA" angle="2.176477937114489" k="236.22864"/>
    <Angle class1="CX" class2="CX" class3="HA" angle="2.261405658516532" k="282.58736"/>
    <Angle class1="NA" class2="CX" class3="HA" angle="2.155499079505517" k="292.62896"/>
    <Angle class1="CT" class2="CT" class3="CV" angle="1.976759910808778" k="807.42832"/>
    <Angle class1="CV" class2="CT" class3="HC" angle="1.905375944402209" k="424.676"/>
    <Angle class1="CT" class2="CV" class3="NB" angle="2.136248097856019" k="507.1008"/>
    <Angle class1="CT" class2="CV" class3="CW" angle="2.234527588035820" k="469.69584"/>
    <Angle class1="NB" class2="CV" class3="CW" angle="1.911624223124349" k="492.70784"/>
    <Angle class1="CV" class2="NB" class3="CR" angle="1.853574572203018" k="1008.59504"/>
    <Angle class1="NB" class2="CR" class3="NA" angle="1.939601851033818" k="467.10176"/>
    <Angle class1="NB" class2="CR" class3="HA" angle="2.198765791662456" k="223.844"/>
    <Angle class1="CR" class2="NA" class3="CW" angle="1.875722800410826" k="529.6944"/>
    <Angle class1="CW" class2="NA" class3="H" angle="2.195257679865948" k="248.02752"/>
    <Angle class1="CV" class2="CW" class3="NA" angle="1.844202154119808" k="505.34352"/>
    <Angle class1="CV" class2="CW" class3="HA" angle="2.295107966372544" k="269.36592"/>
    <Angle class1="NA" class2="CW" class3="HA" angle="2.123629367364100" k="265.76768"/>
    <Angle class1="CT" class2="CT" class3="N3" angle="1.933842264502237" k="939.39168"/>
    <Angle class1="N3" class2="CT" class3="HC" angle="1.856087846325890" k="613.62544"/>
    <Angle class1="CT" class2="N3" class3="H3" angle="1.940491968952335" k="355.2216"/>
    <Angle class1="H3" class2="N3" class3="H3" angle="1.879771964275453" k="307.35664"/>
    <Angle class1="CT" class2="CT" class3="S" angle="2.048492943065745" k="672.53616"/>
    <Angle class1="S" class2="CT" class3="HC" angle="1.888795316508263" k="306.18512"/>
    <Angle class1="CT" class2="S" class3="CT" angle="1.758052702241368" k="1486.82624"/>
    <Angle class1="CT" class2="CT" class3="CA" angle="1.971803175733114" k="809.85504"/>
    <Angle class1="CA" class2="CT" class3="HC" angle="1.927995411508056" k="416.97744"/>
    <Angle class1="CT" class2="CA" class3="CA" angle="2.108811522014669" k="614.2112"/>
    <Angle class1="CA" class2="CA" class3="CA" angle="2.101289152938573" k="807.9304"/>
    <Angle class1="CA" class2="CA" class3="HA" angle="2.091393136079765" k="258.06912"/>
    <Angle class1="CT" class2="N" class3="CT" angle="1.956985330383682" k="557.47616"/>
    <Angle class1="CT" class2="CT" class3="OH" angle="1.902461244551379" k="955.70928"/>
    <Angle class1="OH" class2="CT" class3="HC" angle="1.927890691752936" k="409.11152"/>
    <Angle class1="CT" class2="OH" class3="HO" angle="1.894223290481966" k="545.92832"/>
    <Angle class1="CT" class2="CT" class3="CS" angle="1.972536214018951" k="701.82416"/>
    <Angle class1="CS" class2="CT" class3="HC" angle="1.927332186392298" k="355.97472"/>
    <Angle class1="CT" class2="CS" class3="CW" angle="2.225050450197491" k="489.44432"/>
    <Angle class1="CT" class2="CS" class3="CB" angle="2.201505958588088" k="509.44384"/>
    <Angle class1="CW" class2="CS" class3="CB" angle="1.854935929019573" k="545.34256"/>
    <Angle class1="CS" class2="CW" class3="NA" angle="1.921014094500079" k="480.65792"/>
    <Angle class1="CS" class2="CW" class3="HA" angle="2.258298972447983" k="265.76768"/>
    <Angle class1="CW" class2="NA" class3="CN" angle="1.904503279776212" k="555.38416"/>
    <Angle class1="CN" class2="NA" class3="H" angle="2.192465153062757" k="260.91424"/>
    <Angle class1="NA" class2="CN" class3="CA" angle="2.278754231281356" k="657.80848"/>
    <Angle class1="NA" class2="CN" class3="CB" angle="1.871656183253679" k="637.6416"/>
    <Angle class1="CA" class2="CN" class3="CB" angle="2.132757439352031" k="806.50784"/>
    <Angle class1="CN" class2="CA" class3="CA" angle="2.049470327446862" k="608.18624"/>
    <Angle class1="CN" class2="CA" class3="HA" angle="2.119562750206954" k="245.85184"/>
    <Angle class1="CA" class2="CA" class3="CB" angle="2.074009656729902" k="607.76784"/>
    <Angle class1="CB" class2="CA" class3="HA" angle="2.102825042680328" k="245.34976"/>
    <Angle class1="CS" class2="CB" class3="CN" angle="1.87263356763476" k="644.7544"/>
    <Angle class1="CS" class2="CB" class3="CA" angle="2.330555603480548" k="661.74144"/>
    <Angle class1="CN" class2="CB" class3="CA" angle="2.079961229479202" k="727.5976"/>
    <Angle class1="CA" class2="CA" class3="OH" angle="2.096541857373149" k="730.44272"/>
    <Angle class1="CA" class2="OH" class3="HO" angle="1.917593249166170" k="565.59312"/>
    <Angle class1="S" class2="S" class3="CT" angle="1.781038688490134" k="832.616"/>
    <Angle class1="HT" class2="OT" class3="HT" angle="1.824218134184473" k="460.24"/>
    <!-- THIS IS OPLS PARAMETER -->
    <Angle class1="CW" class2="CV" class3="HA" angle="2.23751210105673" k="292.88"/>
    <Angle class1="NB" class2="CV" class3="HA" angle="2.0943951023932" k="292.88"/>
    <Angle class1="CW" class2="CT" class3="CT" angle="1.9896753472735358" k="527.184"/>
    <Angle class1="CT" class2="CW" class3="NA" angle="2.1223203704251046" k="585.76"/>
    <Angle class1="CV" class2="CW" class3="CT" angle="2.2811453323565885" k="585.76"/>
    <Angle class1="CW" class2="CT" class3="HC" angle="1.911135530933791" k="292.88"/>
  </HarmonicAngleForce>
  <PeriodicTorsionForce>
    <Proper type1="C157" type2="C293" type3="C235" type4="N239" k1="3.721668" k2="0.876548" k3="-0.23012" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C135" type2="C136" type3="C137" type4="C135" k1="0.2092" k2="-0.29288" k3="0.27196" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C135" type2="C136" type3="C137" type4="C224I" k1="0.2092" k2="-0.29288" k3="0.27196" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C135" type2="C136" type3="C137" type4="C283" k1="2.7196" k2="-0.4184" k3="0.4184" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C135" type2="C136" type3="C137" type4="C293" k1="2.7196" k2="-0.4184" k3="0.4184" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C135" type2="C136" type3="C137" type4="H140" k1="0" k2="0" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C135" type2="C137" type3="C135" type4="H140" k1="0" k2="0" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C135" type2="C137" type3="C136" type4="C224" k1="-2.28028" k2="1.48532" k3="0.4184" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C135" type2="C137" type3="C136" type4="C283" k1="2.7196" k2="-0.4184" k3="0.4184" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C135" type2="C137" type3="C136" type4="C293" k1="2.7196" k2="-0.4184" k3="0.4184" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C135" type2="C137" type3="C136" type4="H140" k1="0" k2="0" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C135" type2="C137" type3="C224I" type4="C235" k1="10.48092" k2="-0.39748" k3="2.07108" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C135" type2="C137" type3="C224" type4="C235" k1="-3.64008" k2="-0.23012" k3="0.06276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C135" type2="C137" type3="C224" type4="H140" k1="0" k2="0" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C135" type2="C137" type3="C224I" type4="H140" k1="0" k2="0" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C135" type2="C137" type3="C224" type4="N238" k1="7.97052" k2="-1.1506" k3="1.65268" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C135" type2="C137" type3="C224I" type4="N238" k1="15.71092" k2="-1.23428" k3="-0.43932" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C135" type2="C137" type3="C283" type4="C271" k1="-2.974824" k2="2.234256" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C135" type2="C137" type3="C283" type4="H140" k1="0" k2="0" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C135" type2="C137" type3="C283" type4="N238" k1="6.263448" k2="0.527184" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C135" type2="C137" type3="C293" type4="C235" k1="-2.974824" k2="2.234256" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C135" type2="C137" type3="C293" type4="H140" k1="0" k2="0" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C135" type2="C137" type3="C293" type4="N287" k1="6.263448" k2="0.527184" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C135" type2="C158" type3="C224S" type4="C235" k1="-8.09604" k2="6.52704" k3="-0.1046" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C135" type2="C158" type3="C224S" type4="H140" k1="0" k2="0" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C135" type2="C158" type3="C224S" type4="N238" k1="-6.98728" k2="2.46856" k3="-3.59824" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C135" type2="C158" type3="C283" type4="C271" k1="-2.974824" k2="2.234256" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C135" type2="C158" type3="C283" type4="H140" k1="0" k2="0" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C135" type2="C158" type3="C283" type4="N238" k1="6.263448" k2="0.527184" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C135" type2="C158" type3="C293" type4="C235" k1="-2.974824" k2="2.234256" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C135" type2="C158" type3="C293" type4="H140" k1="0" k2="0" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C135" type2="C158" type3="C293" type4="N287" k1="6.263448" k2="0.527184" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C135" type2="C158" type3="O154" type4="H155" k1="0.43932" k2="-2.53132" k3="0.85772" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C135" type2="C224A" type3="C235" type4="N238" k1="-0.58576" k2="-0.71128" k3="-0.54392" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C135" type2="C224A" type3="C235" type4="N239" k1="-0.54392" k2="-0.66944" k3="-0.50208" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C135" type2="C224A" type3="C235" type4="O236" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C135" type2="C224A" type3="C267" type4="O268" k1="2.092" k2="1.142232" k3="0.9414" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C135" type2="C224A" type3="C267" type4="O269" k1="0" k2="1.142232" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C135" type2="C224A" type3="N238" type4="C235" k1="-0.79496" k2="-0.35564" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C135" type2="C224A" type3="N238" type4="H241" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C135" type2="C235" type3="N238" type4="C223" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C135" type2="C235" type3="N238" type4="C224" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C135" type2="C235" type3="N238" type4="C224A" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C135" type2="C235" type3="N238" type4="C224S" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C135" type2="C235" type3="N238" type4="C224K" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C135" type2="C235" type3="N238" type4="C224Y" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C135" type2="C235" type3="N238" type4="H241" k1="0" k2="10.2508" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C135" type2="C235" type3="N239" type4="C245" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C135" type2="C235" type3="N239" type4="C246" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C135" type2="C283" type3="C271" type4="O272" k1="0" k2="1.71544" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C135" type2="C283" type3="N238" type4="C235" k1="-1.426744" k2="0.27196" k3="0.707096" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C135" type2="C283" type3="N238" type4="H241" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C135" type2="C293" type3="C235" type4="N238" k1="3.721668" k2="0.876548" k3="-0.23012" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C135" type2="C293" type3="C235" type4="N239" k1="3.721668" k2="0.876548" k3="-0.23012" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C135" type2="C293" type3="C235" type4="O236" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C135" type2="C293" type3="N287" type4="H290" k1="0" k2="0" k3="0.725924" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C136" type2="C136" type3="C136" type4="C224K" k1="-5.23" k2="-1.00416" k3="0.79496" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C267" type2="C136" type3="C136" type4="C224" k1="2.7196" k2="-0.4184" k3="0.4184" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C136" type2="C136" type3="C136" type4="C283" k1="2.7196" k2="-0.4184" k3="0.4184" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C136" type2="C136" type3="C136" type4="C292" k1="-5.23" k2="-1.00416" k3="0.79496" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C136" type2="C136" type3="C136" type4="C293" k1="2.7196" k2="-0.4184" k3="0.4184" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C136" type2="C136" type3="C136" type4="H140" k1="0" k2="0" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C267" type2="C136" type3="C136" type4="H140" k1="0" k2="0" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C136" type2="C136" type3="C224" type4="C235" k1="-3.38904" k2="0.50208" k3="0.16736" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C136" type2="C136" type3="C224K" type4="C235" k1="-4.184" k2="-0.4184" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C136" type2="C136" type3="C224K" type4="H140" k1="0" k2="0" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C136" type2="C136" type3="C224" type4="H140" k1="0" k2="0" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C136" type2="C136" type3="C224" type4="N238" k1="4.45596" k2="-0.77404" k3="3.30536" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C136" type2="C136" type3="C224K" type4="N238" k1="0.1046" k2="-0.27196" k3="1.82004" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C136" type2="C136" type3="C235" type4="N237" k1="5.949648" k2="-0.755212" k3="-0.6799" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C136" type2="C136" type3="C235" type4="O236" k1="0.849352" k2="2.727968" k3="0.290788" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C136" type2="C136" type3="C245" type4="H140" k1="0" k2="0" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C136" type2="C136" type3="C245" type4="N239" k1="1.76774" k2="-2.012504" k3="1.491596" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C136" type2="C136" type3="C246" type4="C235" k1="1.10876" k2="0.71128" k3="0.25104" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C136" type2="C136" type3="C246" type4="H140" k1="0" k2="0" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C136" type2="C136" type3="C246" type4="N239" k1="-4.35136" k2="-1.3598" k3="1.21336" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C136" type2="C136" type3="C267" type4="O268" k1="2.092" k2="1.142232" k3="0.9414" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C136" type2="C136" type3="C267" type4="O269" k1="0" k2="1.142232" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C136" type2="C136" type3="C283" type4="C271" k1="-4.932936" k2="1.905812" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C136" type2="C136" type3="C283" type4="H140" k1="0" k2="0" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C136" type2="C136" type3="C283" type4="N238" k1="1.849328" k2="1.876524" k3="1.84096" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C136" type2="C136" type3="C285" type4="C271" k1="-3.663092" k2="3.359752" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C136" type2="C136" type3="C285" type4="H140" k1="0" k2="0" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C136" type2="C136" type3="C285" type4="N239" k1="3.288624" k2="0.332628" k3="0.4184" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C136" type2="C136" type3="C292" type4="H140" k1="0" k2="0" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C136" type2="C136" type3="C292" type4="N287" k1="5.715344" k2="-0.479068" k3="1.01462" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C136" type2="C136" type3="C293" type4="C235" k1="-4.932936" k2="1.905812" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C136" type2="C136" type3="C293" type4="H140" k1="0" k2="0" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C136" type2="C136" type3="C293" type4="N287" k1="1.849328" k2="1.876524" k3="1.84096" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C136" type2="C136" type3="C295" type4="C235" k1="-3.663092" k2="3.359752" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C136" type2="C136" type3="C295" type4="H140" k1="0" k2="0" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C136" type2="C136" type3="C295" type4="N309" k1="5.715344" k2="-0.479068" k3="1.01462" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C136" type2="C136" type3="C296" type4="H140" k1="0" k2="0" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C136" type2="C136" type3="C296" type4="N309" k1="5.715344" k2="-0.479068" k3="1.01462" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C136" type2="C137" type3="C135" type4="H140" k1="0" k2="0" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C136" type2="C137" type3="C224I" type4="C235" k1="10.48092" k2="-0.39748" k3="2.07108" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C136" type2="C137" type3="C224I" type4="H140" k1="0" k2="0" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C136" type2="C137" type3="C224I" type4="N238" k1="15.71092" k2="-1.23428" k3="-0.43932" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C136" type2="C137" type3="C283" type4="C271" k1="-2.974824" k2="2.234256" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C136" type2="C137" type3="C283" type4="H140" k1="0" k2="0" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C136" type2="C137" type3="C283" type4="N238" k1="6.263448" k2="0.527184" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C136" type2="C137" type3="C293" type4="C235" k1="-2.974824" k2="2.234256" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C136" type2="C137" type3="C293" type4="H140" k1="0" k2="0" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C136" type2="C137" type3="C293" type4="N287" k1="6.263448" k2="0.527184" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C136" type2="C210" type3="S202" type4="C209" k1="1.9351" k2="-1.204992" k3="1.416284" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C136" type2="C224" type3="C235" type4="N238" k1="-0.54392" k2="-0.66944" k3="-0.50208" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C136" type2="C224K" type3="C235" type4="N238" k1="-0.54392" k2="-0.66944" k3="-0.50208" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C136" type2="C224" type3="C235" type4="N239" k1="-0.54392" k2="-0.66944" k3="-0.50208" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C136" type2="C224K" type3="C235" type4="N239" k1="-0.54392" k2="-0.66944" k3="-0.50208" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C136" type2="C224" type3="C235" type4="O236" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C136" type2="C224K" type3="C235" type4="O236" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C136" type2="C224" type3="N238" type4="C235" k1="-0.77404" k2="-0.33472" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C136" type2="C224K" type3="N238" type4="C235" k1="-0.77404" k2="-0.33472" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C136" type2="C224" type3="N238" type4="H241" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C136" type2="C224K" type3="N238" type4="H241" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C136" type2="C235" type3="N237" type4="H240" k1="0" k2="10.2508" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C136" type2="C245" type3="N239" type4="C235" k1="-0.77404" k2="0.33472" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C136" type2="C245" type3="N239" type4="C246" k1="5.981028" k2="4.305336" k3="-23.568472" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C136" type2="C245" type3="N239" type4="C285" k1="5.981028" k2="4.305336" k3="-23.568472" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C136" type2="C246" type3="C235" type4="N238" k1="3.28444" k2="-2.28028" k3="2.76144" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C136" type2="C246" type3="C235" type4="N239" k1="3.28444" k2="-2.28028" k3="2.76144" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C136" type2="C246" type3="C235" type4="O236" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C136" type2="C246" type3="N239" type4="C235" k1="-0.77404" k2="-0.33472" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C136" type2="C246" type3="N239" type4="C245" k1="9.943276" k2="-1.535528" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C136" type2="C267" type3="O268" type4="H270" k1="3.138" k2="11.506" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C136" type2="C274" type3="C271" type4="O272" k1="0" k2="1.142232" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C136" type2="C283" type3="C271" type4="O272" k1="0" k2="1.71544" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C136" type2="C283" type3="N238" type4="C235" k1="-1.426744" k2="0.27196" k3="0.707096" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C136" type2="C283K" type3="N238" type4="C235" k1="-1.426744" k2="0.27196" k3="0.707096" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C136" type2="C283" type3="N238" type4="H241" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C136" type2="C283K" type3="N238" type4="H241" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C136" type2="C285" type3="C271" type4="O272" k1="0" k2="1.71544" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C136" type2="C285" type3="N239" type4="C235" k1="-1.426744" k2="0.27196" k3="0.707096" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C136" type2="C285" type3="N239" type4="C245" k1="9.943276" k2="-1.535528" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C136" type2="C292" type3="N287" type4="H290" k1="0" k2="0" k3="0.725924" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C136" type2="C293" type3="C235" type4="N238" k1="3.721668" k2="0.876548" k3="-0.23012" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C136" type2="C293" type3="C235" type4="N239" k1="3.721668" k2="0.876548" k3="-0.23012" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C136" type2="C293" type3="C235" type4="O236" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C136" type2="C293" type3="N287" type4="H290" k1="0" k2="0" k3="0.725924" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C136" type2="C295" type3="C235" type4="N238" k1="10.520668" k2="1.504148" k3="4.68608" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C136" type2="C295" type3="C235" type4="N239" k1="10.520668" k2="1.504148" k3="4.68608" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C136" type2="C295" type3="C235" type4="O236" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C136" type2="C295" type3="N309" type4="C296" k1="3.0080868" k2="-0.2589896" k3="0.5520788" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C136" type2="C295" type3="N309" type4="H310" k1="0" k2="0" k3="0.725924" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C136" type2="C296" type3="N309" type4="C295" k1="3.0080868" k2="-0.2589896" k3="0.5520788" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C136" type2="C296" type3="N309" type4="H310" k1="0" k2="0" k3="0.725924" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C136" type2="C308" type3="C307" type4="H140" k1="0" k2="0" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C136" type2="C308" type3="C307" type4="N303" k1="5.715344" k2="-0.479068" k3="1.01462" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C137" type2="C136" type3="C135" type4="H140" k1="0" k2="0" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C137" type2="C136" type3="C224" type4="C235" k1="1.10876" k2="0.71128" k3="0.25104" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C137" type2="C136" type3="C224" type4="H140" k1="0" k2="0" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C137" type2="C136" type3="C224" type4="N238" k1="-4.35136" k2="-1.3598" k3="1.21336" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C137" type2="C136" type3="C283" type4="C271" k1="-3.663092" k2="3.359752" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C137" type2="C136" type3="C283" type4="H140" k1="0" k2="0" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C137" type2="C136" type3="C283" type4="N238" k1="3.288624" k2="0.332628" k3="0.4184" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C137" type2="C136" type3="C293" type4="C235" k1="-3.663092" k2="3.359752" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C137" type2="C136" type3="C293" type4="H140" k1="0" k2="0" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C137" type2="C136" type3="C293" type4="N287" k1="3.288624" k2="0.332628" k3="0.4184" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C137" type2="C224" type3="C235" type4="N238" k1="-0.54392" k2="-0.66944" k3="-0.50208" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C137" type2="C224I" type3="C235" type4="N238" k1="-0.54392" k2="-0.66944" k3="-0.50208" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C137" type2="C224" type3="C235" type4="N239" k1="-0.54392" k2="-0.66944" k3="-0.50208" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C137" type2="C224I" type3="C235" type4="N239" k1="-0.54392" k2="-0.66944" k3="-0.50208" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C137" type2="C224" type3="C235" type4="O236" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C137" type2="C224I" type3="C235" type4="O236" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C137" type2="C224" type3="N238" type4="C235" k1="-0.77404" k2="-0.33472" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C137" type2="C224I" type3="N238" type4="C235" k1="-0.77404" k2="-0.33472" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C137" type2="C224" type3="N238" type4="H241" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C137" type2="C224I" type3="N238" type4="H241" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C137" type2="C283" type3="C271" type4="O272" k1="0" k2="1.71544" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C137" type2="C283" type3="N238" type4="C235" k1="-1.426744" k2="0.27196" k3="0.707096" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C137" type2="C283" type3="N238" type4="H241" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C137" type2="C293" type3="C235" type4="N238" k1="3.721668" k2="0.876548" k3="-0.23012" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C137" type2="C293" type3="C235" type4="N239" k1="3.721668" k2="0.876548" k3="-0.23012" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C137" type2="C293" type3="C235" type4="O236" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C137" type2="C293" type3="N287" type4="H290" k1="0" k2="0" k3="0.725924" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C145" type2="C145" type3="C149" type4="C224" k1="-10.89932" k2="0.39748" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C145" type2="C145" type3="C149" type4="C224Y" k1="-9.2048" k2="0.39748" k3="1.23428" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C145" type2="C145" type3="C149" type4="C283" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C145" type2="C145" type3="C149" type4="C293" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C145" type2="C145" type3="C149" type4="H140" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C145" type2="C149" type3="C224" type4="C235" k1="0.96232" k2="1.21336" k3="0.66944" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C145" type2="C149" type3="C224Y" type4="C235" k1="-0.27196" k2="2.13384" k3="-0.02092" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C145" type2="C149" type3="C224" type4="H140" k1="0" k2="0" k3="0.966504" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C145" type2="C149" type3="C224Y" type4="H140" k1="0" k2="0" k3="0.966504" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C145" type2="C149" type3="C224" type4="N238" k1="-1.96648" k2="1.58992" k3="1.90372" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C145" type2="C149" type3="C224Y" type4="N238" k1="-0.25104" k2="1.4644" k3="2.23844" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C145" type2="C149" type3="C283" type4="C271" k1="-2.941352" k2="3.717484" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C145" type2="C149" type3="C283" type4="H140" k1="0" k2="0" k3="0.966504" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C145" type2="C149" type3="C283" type4="N238" k1="3.581504" k2="1.5167" k3="0.765672" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C145" type2="C149" type3="C293" type4="C235" k1="-2.941352" k2="3.717484" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C145" type2="C149" type3="C293" type4="H140" k1="0" k2="0" k3="0.966504" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C145" type2="C149" type3="C293" type4="N287" k1="3.581504" k2="1.5167" k3="0.765672" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C145" type2="C166" type3="C145" type4="H146" k1="0" k2="15.167" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C145" type2="C166" type3="O167" type4="H168" k1="0" k2="3.518744" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C145" type2="C501" type3="C500" type4="C514" k1="0" k2="7.0082" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C145" type2="C502" type3="C501" type4="C500" k1="0" k2="12.552" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C149" type2="C224" type3="C235" type4="N238" k1="-0.54392" k2="-0.66944" k3="-0.50208" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C149" type2="C224Y" type3="C235" type4="N238" k1="-0.54392" k2="-0.66944" k3="-0.50208" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C149" type2="C224" type3="C235" type4="N239" k1="-0.54392" k2="-0.66944" k3="-0.50208" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C149" type2="C224Y" type3="C235" type4="N239" k1="-0.54392" k2="-0.66944" k3="-0.50208" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C149" type2="C224" type3="C235" type4="O236" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C149" type2="C224Y" type3="C235" type4="O236" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C149" type2="C224" type3="N238" type4="C235" k1="-0.77404" k2="-0.33472" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C149" type2="C224Y" type3="N238" type4="C235" k1="-0.77404" k2="-0.33472" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C149" type2="C224" type3="N238" type4="H241" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C149" type2="C224Y" type3="N238" type4="H241" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C149" type2="C283" type3="C271" type4="O272" k1="0" k2="1.71544" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C149" type2="C283" type3="N238" type4="C235" k1="-1.426744" k2="0.27196" k3="0.707096" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C149" type2="C283" type3="N238" type4="H241" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C149" type2="C293" type3="C235" type4="N238" k1="3.721668" k2="0.876548" k3="-0.23012" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C149" type2="C293" type3="C235" type4="N239" k1="3.721668" k2="0.876548" k3="-0.23012" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C149" type2="C293" type3="C235" type4="O236" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C149" type2="C293" type3="N287" type4="H290" k1="0" k2="0" k3="0.725924" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C157" type2="C224S" type3="C235" type4="N238" k1="0.35564" k2="0.25104" k3="0.25104" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C157" type2="C224S" type3="C235" type4="N239" k1="0.35564" k2="0.25104" k3="0.25104" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C157" type2="C224S" type3="C235" type4="O236" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C157" type2="C224S" type3="N238" type4="C235" k1="-0.04184" k2="0.39748" k3="-0.25104" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C157" type2="C224S" type3="N238" type4="H241" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C157" type2="C283" type3="C271" type4="O272" k1="0" k2="1.71544" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C157" type2="C283" type3="N238" type4="C235" k1="-1.426744" k2="0.27196" k3="0.707096" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C157" type2="C283" type3="N238" type4="H241" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C157" type2="C293" type3="C235" type4="N238" k1="3.721668" k2="0.876548" k3="-0.23012" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C157" type2="C293" type3="C235" type4="O236" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C157" type2="C293" type3="N287" type4="H290" k1="0" k2="0" k3="0.725924" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C158" type2="C224S" type3="C235" type4="N238" k1="0.35564" k2="0.25104" k3="0.25104" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C158" type2="C224S" type3="C235" type4="N239" k1="0.35564" k2="0.25104" k3="0.25104" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C158" type2="C224S" type3="C235" type4="O236" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C158" type2="C224S" type3="N238" type4="C235" k1="-0.04184" k2="0.39748" k3="-0.25104" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C158" type2="C224S" type3="N238" type4="H241" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C158" type2="C283" type3="C271" type4="O272" k1="0" k2="1.71544" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C158" type2="C283" type3="N238" type4="C235" k1="-1.426744" k2="0.27196" k3="0.707096" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C158" type2="C283" type3="N238" type4="H241" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C158" type2="C293" type3="C235" type4="N238" k1="3.721668" k2="0.876548" k3="-0.23012" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C158" type2="C293" type3="C235" type4="N239" k1="3.721668" k2="0.876548" k3="-0.23012" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C158" type2="C293" type3="C235" type4="O236" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C158" type2="C293" type3="N287" type4="H290" k1="0" k2="0" k3="0.725924" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C206" type2="C224" type3="C235" type4="N238" k1="-0.54392" k2="-0.66944" k3="-0.50208" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C206" type2="C224" type3="C235" type4="N239" k1="-0.54392" k2="-0.66944" k3="-0.50208" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C206" type2="C224" type3="C235" type4="O236" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C206" type2="C224" type3="N238" type4="C235" k1="-0.77404" k2="-0.33472" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C206" type2="C224" type3="N238" type4="H241" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C206" type2="C283" type3="C271" type4="O272" k1="0" k2="1.71544" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C206" type2="C283" type3="N238" type4="C235" k1="-1.426744" k2="0.27196" k3="0.707096" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C206" type2="C283" type3="N238" type4="H241" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C206" type2="C293" type3="C235" type4="N238" k1="3.721668" k2="0.876548" k3="-0.23012" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C206" type2="C293" type3="C235" type4="N239" k1="3.721668" k2="0.876548" k3="-0.23012" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C206" type2="C293" type3="C235" type4="O236" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C206" type2="C293" type3="N287" type4="H290" k1="0" k2="0" k3="0.725924" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C209" type2="S202" type3="C210" type4="H140" k1="0" k2="0" k3="1.353524" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C210" type2="C136" type3="C224" type4="C235" k1="1.69452" k2="0.06276" k3="3.40996" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C210" type2="C136" type3="C224" type4="H140" k1="0" k2="0" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C210" type2="C136" type3="C224" type4="N238" k1="0.39748" k2="-1.73636" k3="-1.54808" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C210" type2="C136" type3="C283" type4="C271" k1="-1.905812" k2="1.462308" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C210" type2="C136" type3="C283" type4="H140" k1="0" k2="0" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C210" type2="C136" type3="C283" type4="N238" k1="0.447688" k2="1.131772" k3="0.820064" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C210" type2="C136" type3="C293" type4="C235" k1="-1.905812" k2="1.462308" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C210" type2="C136" type3="C293" type4="H140" k1="0" k2="0" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C210" type2="C136" type3="C293" type4="N287" k1="0.447688" k2="1.131772" k3="0.820064" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C210" type2="S202" type3="C209" type4="H140" k1="0" k2="0" k3="1.353524" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C214" type2="C224" type3="C235" type4="N238" k1="-0.54392" k2="-0.66944" k3="-0.50208" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C214" type2="C224" type3="C235" type4="N239" k1="-0.54392" k2="-0.66944" k3="-0.50208" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C214" type2="C224" type3="C235" type4="O236" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C214" type2="C224" type3="N238" type4="C235" k1="-0.77404" k2="-0.33472" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C214" type2="C224" type3="N238" type4="H241" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C214" type2="C283" type3="C271" type4="O272" k1="0" k2="1.71544" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C214" type2="C283" type3="N238" type4="C235" k1="-1.426744" k2="0.27196" k3="0.707096" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C214" type2="C283" type3="N238" type4="H241" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C214" type2="C293" type3="C235" type4="N238" k1="3.721668" k2="0.876548" k3="-0.23012" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C214" type2="C293" type3="C235" type4="N239" k1="3.721668" k2="0.876548" k3="-0.23012" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C214" type2="C293" type3="C235" type4="O236" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C214" type2="C293" type3="N287" type4="H290" k1="0" k2="0" k3="0.725924" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C214" type2="S203" type3="S203" type4="C214" k1="0" k2="-15.510088" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C223" type2="C235" type3="N238" type4="C223" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C223" type2="C235" type3="N238" type4="C224" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C223" type2="C235" type3="N238" type4="C224I" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C223" type2="C235" type3="N238" type4="C224A" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C223" type2="C235" type3="N238" type4="C224K" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C223" type2="C235" type3="N238" type4="C224S" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C223" type2="C235" type3="N238" type4="C224Y" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C223" type2="C235" type3="N238" type4="C242" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C223" type2="C235" type3="N238" type4="C283" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C295" type2="C235" type3="N238" type4="C283" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C223" type2="C235" type3="N238" type4="C284" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C223" type2="C235" type3="N238" type4="H241" k1="0" k2="10.2508" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C223" type2="C235" type3="N239" type4="C245" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C223" type2="C235" type3="N239" type4="C246" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C223" type2="C235" type3="N239" type4="C285" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C223" type2="N238" type3="C235" type4="C224" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C223" type2="N238" type3="C235" type4="C224I" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C223" type2="N238" type3="C235" type4="C224A" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C223" type2="N238" type3="C235" type4="C224S" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C223" type2="N238" type3="C235" type4="C224K" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C223" type2="N238" type3="C235" type4="C224Y" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C223" type2="N238" type3="C235" type4="C246" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C223" type2="N238" type3="C235" type4="C292" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C223" type2="N238" type3="C235" type4="C293" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C223" type2="N238" type3="C235" type4="C295" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C223" type2="N238" type3="C235" type4="O236" k1="0" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224" type2="C136" type3="C136" type4="C235" k1="-2.42672" k2="2.11292" k3="-0.12552" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224K" type2="C136" type3="C136" type4="H140" k1="0" k2="0" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224" type2="C136" type3="C136" type4="H140" k1="0" k2="0" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224" type2="C136" type3="C137" type4="H140" k1="0" k2="0" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224" type2="C136" type3="C210" type4="H140" k1="0" k2="0" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224" type2="C136" type3="C210" type4="S202" k1="-2.4058" k2="-0.9414" k3="-0.48116" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224" type2="C136" type3="C235" type4="N237" k1="1.69452" k2="1.06692" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224" type2="C136" type3="C235" type4="O236" k1="3.32628" k2="-0.2092" k3="-0.2092" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224" type2="C136" type3="C267" type4="O268" k1="2.092" k2="1.142232" k3="0.9414" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224" type2="C136" type3="C267" type4="O269" k1="0" k2="1.142232" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224" type2="C136" type3="C274" type4="C271" k1="-18.43052" k2="-0.23012" k3="-1.6736" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224" type2="C136" type3="C274" type4="H140" k1="0" k2="0" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224" type2="C136" type3="C308" type4="C307" k1="-2.1966" k2="-0.523" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224" type2="C136" type3="C308" type4="H140" k1="0" k2="0" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224" type2="C136" type3="C500" type4="C501" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224" type2="C136" type3="C500" type4="C514" k1="1.00416" k2="-1.21336" k3="-0.39748" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224" type2="C137" type3="C135" type4="H140" k1="0" k2="0" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224I" type2="C137" type3="C135" type4="H140" k1="0" k2="0" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224I" type2="C137" type3="C136" type4="H140" k1="0" k2="0" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224S" type2="C157" type3="O154" type4="H155" k1="0.43932" k2="-2.53132" k3="0.85772" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224S" type2="C158" type3="C135" type4="H140" k1="0" k2="0" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224S" type2="C158" type3="O154" type4="H155" k1="0.43932" k2="-2.53132" k3="0.85772" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224" type2="C206" type3="S200" type4="H204" k1="1.63176" k2="2.17568" k3="4.37228" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224" type2="C214" type3="S203" type4="S203" k1="4.060572" k2="-1.748912" k3="1.95602" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224" type2="C235" type3="N238" type4="C224" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224" type2="C235" type3="N238" type4="C224I" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224" type2="C235" type3="N238" type4="C224A" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224" type2="C235" type3="N238" type4="C224S" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224" type2="C235" type3="N238" type4="C224K" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224" type2="C235" type3="N238" type4="C224Y" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224I" type2="C235" type3="N238" type4="C224" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224I" type2="C235" type3="N238" type4="C224I" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224I" type2="C235" type3="N238" type4="C224A" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224I" type2="C235" type3="N238" type4="C224S" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224I" type2="C235" type3="N238" type4="C224K" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224I" type2="C235" type3="N238" type4="C224Y" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224A" type2="C235" type3="N238" type4="C224" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224A" type2="C235" type3="N238" type4="C224I" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224A" type2="C235" type3="N238" type4="C224A" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224A" type2="C235" type3="N238" type4="C224S" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224A" type2="C235" type3="N238" type4="C224K" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224A" type2="C235" type3="N238" type4="C224Y" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224S" type2="C235" type3="N238" type4="C224" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224S" type2="C235" type3="N238" type4="C224I" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224S" type2="C235" type3="N238" type4="C224A" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224S" type2="C235" type3="N238" type4="C224S" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224S" type2="C235" type3="N238" type4="C224K" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224S" type2="C235" type3="N238" type4="C224Y" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224K" type2="C235" type3="N238" type4="C224" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224K" type2="C235" type3="N238" type4="C224I" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224K" type2="C235" type3="N238" type4="C224A" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224K" type2="C235" type3="N238" type4="C224S" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224K" type2="C235" type3="N238" type4="C224K" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224K" type2="C235" type3="N238" type4="C224Y" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224Y" type2="C235" type3="N238" type4="C224" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224Y" type2="C235" type3="N238" type4="C224I" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224Y" type2="C235" type3="N238" type4="C224A" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224Y" type2="C235" type3="N238" type4="C224S" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224Y" type2="C235" type3="N238" type4="C224K" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224Y" type2="C235" type3="N238" type4="C224Y" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224" type2="C235" type3="N238" type4="C242" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224I" type2="C235" type3="N238" type4="C242" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224A" type2="C235" type3="N238" type4="C242" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224S" type2="C235" type3="N238" type4="C242" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224K" type2="C235" type3="N238" type4="C242" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224Y" type2="C235" type3="N238" type4="C242" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224" type2="C235" type3="N238" type4="C283" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224I" type2="C235" type3="N238" type4="C283" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224A" type2="C235" type3="N238" type4="C283" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224S" type2="C235" type3="N238" type4="C283" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224K" type2="C235" type3="N238" type4="C283" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224Y" type2="C235" type3="N238" type4="C283" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224" type2="C235" type3="N238" type4="C284" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224A" type2="C235" type3="N238" type4="C284" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224S" type2="C235" type3="N238" type4="C284" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224K" type2="C235" type3="N238" type4="C284" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224Y" type2="C235" type3="N238" type4="C284" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224" type2="C235" type3="N238" type4="H241" k1="0" k2="10.2508" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224I" type2="C235" type3="N238" type4="H241" k1="0" k2="10.2508" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224A" type2="C235" type3="N238" type4="H241" k1="0" k2="10.2508" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224S" type2="C235" type3="N238" type4="H241" k1="0" k2="10.2508" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224K" type2="C235" type3="N238" type4="H241" k1="0" k2="10.2508" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224Y" type2="C235" type3="N238" type4="H241" k1="0" k2="10.2508" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224" type2="C235" type3="N239" type4="C245" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224I" type2="C235" type3="N239" type4="C245" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224A" type2="C235" type3="N239" type4="C245" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224S" type2="C235" type3="N239" type4="C245" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224K" type2="C235" type3="N239" type4="C245" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224Y" type2="C235" type3="N239" type4="C245" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224" type2="C235" type3="N239" type4="C246" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224I" type2="C235" type3="N239" type4="C246" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224A" type2="C235" type3="N239" type4="C246" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224S" type2="C235" type3="N239" type4="C246" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224K" type2="C235" type3="N239" type4="C246" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224Y" type2="C235" type3="N239" type4="C246" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224" type2="C235" type3="N239" type4="C285" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224I" type2="C235" type3="N239" type4="C285" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224A" type2="C235" type3="N239" type4="C285" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224S" type2="C235" type3="N239" type4="C285" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224K" type2="C235" type3="N239" type4="C285" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224Y" type2="C235" type3="N239" type4="C285" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C223" type2="C267" type3="O268" type4="H270" k1="3.138" k2="11.506" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224" type2="C267" type3="O268" type4="H270" k1="3.138" k2="11.506" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224I" type2="C267" type3="O268" type4="H270" k1="3.138" k2="11.506" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224S" type2="C267" type3="O268" type4="H270" k1="3.138" k2="11.506" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224S" type2="C267" type3="O268" type4="H270" k1="3.138" k2="11.506" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224K" type2="C267" type3="O268" type4="H270" k1="3.138" k2="11.506" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224Y" type2="C267" type3="O268" type4="H270" k1="3.138" k2="11.506" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224" type2="C274" type3="C271" type4="O272" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224" type2="C505" type3="C507" type4="C508" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224" type2="C505" type3="C507" type4="N511" k1="-2.48948" k2="-1.046" k3="0.06276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224" type2="C505" type3="C508" type4="C507" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224" type2="C505" type3="C508" type4="N503" k1="6.08772" k2="-3.36812" k3="0.8368" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224" type2="C505" type3="C510" type4="C510" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224" type2="C505" type3="C510" type4="N512" k1="-0.79496" k2="1.046" k3="0.39748" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224" type2="N238" type3="C235" type4="C246" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224I" type2="N238" type3="C235" type4="C246" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224A" type2="N238" type3="C235" type4="C246" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224S" type2="N238" type3="C235" type4="C246" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224K" type2="N238" type3="C235" type4="C246" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224Y" type2="N238" type3="C235" type4="C246" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224" type2="N238" type3="C235" type4="C292" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224I" type2="N238" type3="C235" type4="C292" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224A" type2="N238" type3="C235" type4="C292" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224S" type2="N238" type3="C235" type4="C292" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224K" type2="N238" type3="C235" type4="C292" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224Y" type2="N238" type3="C235" type4="C292" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224" type2="N238" type3="C235" type4="C292" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224" type2="N238" type3="C235" type4="C293" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224I" type2="N238" type3="C235" type4="C293" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224A" type2="N238" type3="C235" type4="C293" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224S" type2="N238" type3="C235" type4="C293" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224K" type2="N238" type3="C235" type4="C293" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224Y" type2="N238" type3="C235" type4="C293" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224" type2="N238" type3="C235" type4="C295" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224I" type2="N238" type3="C235" type4="C295" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224A" type2="N238" type3="C235" type4="C295" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224S" type2="N238" type3="C235" type4="C295" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224K" type2="N238" type3="C235" type4="C295" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224Y" type2="N238" type3="C235" type4="C295" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224" type2="N238" type3="C235" type4="O236" k1="0" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224I" type2="N238" type3="C235" type4="O236" k1="0" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224A" type2="N238" type3="C235" type4="O236" k1="0" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224S" type2="N238" type3="C235" type4="O236" k1="0" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224K" type2="N238" type3="C235" type4="O236" k1="0" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C224Y" type2="N238" type3="C235" type4="O236" k1="0" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C235" type2="C136" type3="C136" type4="C283" k1="-2.650564" k2="1.002068" k3="-1.016712" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C235" type2="C136" type3="C136" type4="C293" k1="-2.650564" k2="1.002068" k3="-1.016712" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C235" type2="C136" type3="C136" type4="H140" k1="0" k2="0" k3="-0.2092" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C235" type2="C136" type3="C224" type4="C235" k1="10.02068" k2="1.2552" k3="0.43932" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C235" type2="C136" type3="C224" type4="H140" k1="0" k2="0" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C235" type2="C136" type3="C224" type4="N238" k1="-15.71092" k2="1.10876" k3="-1.17152" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C235" type2="C136" type3="C283" type4="C271" k1="1.251016" k2="3.259336" k3="0.53346" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C235" type2="C136" type3="C283" type4="H140" k1="0" k2="0" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C235" type2="C136" type3="C283" type4="N238" k1="-11.508092" k2="3.194484" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C235" type2="C136" type3="C293" type4="C235" k1="1.251016" k2="3.259336" k3="0.53346" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C235" type2="C136" type3="C293" type4="H140" k1="0" k2="0" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C235" type2="C136" type3="C293" type4="N287" k1="-11.508092" k2="3.194484" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C235" type2="C223" type3="N238" type4="C235" k1="0.14644" k2="0.87864" k3="-0.77404" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C235" type2="C223" type3="N238" type4="H241" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C235" type2="C224A" type3="C135" type4="H140" k1="0" k2="0" k3="-0.158992" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C235" type2="C224" type3="C136" type4="C267" k1="1.251016" k2="3.259336" k3="0.53346" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C235" type2="C224" type3="C136" type4="C274" k1="-4.72792" k2="-3.91204" k3="2.17568" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C235" type2="C224" type3="C136" type4="C308" k1="-1.1506" k2="-0.3661" k3="0.46024" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C235" type2="C224" type3="C136" type4="C500" k1="-0.8368" k2="1.8828" k3="0.25104" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C235" type2="C224" type3="C136" type4="H140" k1="0" k2="0" k3="-0.158992" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C235" type2="C224K" type3="C136" type4="H140" k1="0" k2="0" k3="-0.158992" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C235" type2="C224" type3="C137" type4="H140" k1="0" k2="0" k3="-0.158992" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C235" type2="C224I" type3="C137" type4="H140" k1="0" k2="0" k3="-0.158992" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C235" type2="C224" type3="C149" type4="H140" k1="0" k2="0" k3="-0.158992" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C235" type2="C224Y" type3="C149" type4="H140" k1="0" k2="0" k3="-0.158992" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C235" type2="C224S" type3="C157" type4="H140" k1="0" k2="0" k3="-0.158992" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C235" type2="C224S" type3="C157" type4="O154" k1="-14.49756" k2="1.9874" k3="2.44764" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C235" type2="C224S" type3="C158" type4="H140" k1="0" k2="0" k3="-0.158992" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C235" type2="C224S" type3="C158" type4="O154" k1="-3.36812" k2="4.45596" k3="-1.86188" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C235" type2="C224" type3="C206" type4="H140" k1="0" k2="0" k3="-0.158992" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C235" type2="C224" type3="C206" type4="S200" k1="-5.87852" k2="-0.06276" k3="-0.18828" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C235" type2="C224" type3="C214" type4="H140" k1="0" k2="0" k3="-0.158992" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C235" type2="C224" type3="C214" type4="S203" k1="-6.951716" k2="1.106668" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C235" type2="C224" type3="C274" type4="C271" k1="7.48936" k2="1.52716" k3="-0.25104" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C235" type2="C224" type3="C274" type4="H140" k1="0" k2="0" k3="-0.158992" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C235" type2="C224" type3="C505" type4="C507" k1="3.9748" k2="1.046" k3="0.2092" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C235" type2="C224" type3="C505" type4="C508" k1="0.4184" k2="1.2552" k3="0.4184" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C235" type2="C224" type3="C505" type4="C510" k1="-0.58576" k2="-1.90372" k3="0.02092" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C235" type2="C224" type3="C505" type4="H140" k1="0" k2="0" k3="-0.158992" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C235" type2="C224" type3="N238" type4="C235" k1="0.04184" k2="0.85772" k3="-0.18828" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C235" type2="C224I" type3="N238" type4="C235" k1="0.04184" k2="0.85772" k3="-0.18828" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C235" type2="C224K" type3="N238" type4="C235" k1="0.04184" k2="0.85772" k3="-0.18828" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C235" type2="C224Y" type3="N238" type4="C235" k1="0.04184" k2="0.85772" k3="-0.18828" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C235" type2="C224A" type3="N238" type4="C235" k1="0.04184" k2="0.85772" k3="-0.14644" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C235" type2="C224S" type3="N238" type4="C235" k1="0.16736" k2="-0.33472" k3="-0.27196" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C235" type2="C224" type3="N238" type4="H241" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C235" type2="C224I" type3="N238" type4="H241" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C235" type2="C224A" type3="N238" type4="H241" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C235" type2="C224S" type3="N238" type4="H241" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C235" type2="C224K" type3="N238" type4="H241" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C235" type2="C224Y" type3="N238" type4="H241" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C235" type2="N238" type3="C223" type4="C267" k1="-5.253012" k2="0.43932" k3="-0.4184" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C235" type2="N238" type3="C224" type4="C267" k1="-5.253012" k2="0.43932" k3="-0.4184" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C235" type2="N238" type3="C224I" type4="C267" k1="-5.253012" k2="0.43932" k3="-0.4184" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C235" type2="N238" type3="C224A" type4="C267" k1="-5.253012" k2="0.43932" k3="-0.4184" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C235" type2="N238" type3="C224S" type4="C267" k1="-5.253012" k2="0.43932" k3="-0.4184" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C235" type2="N238" type3="C224K" type4="C267" k1="-5.253012" k2="0.43932" k3="-0.4184" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C235" type2="N238" type3="C224Y" type4="C267" k1="-5.253012" k2="0.43932" k3="-0.4184" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C235" type2="C246" type3="C136" type4="H140" k1="0" k2="0" k3="-0.158992" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C235" type2="C246" type3="N239" type4="C235" k1="0.04184" k2="0.85772" k3="-0.18828" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C235" type2="C246" type3="N239" type4="C245" k1="-3.633804" k2="2.617092" k3="-7.324092" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C235" type2="C292" type3="N287" type4="H290" k1="0" k2="0" k3="0.725924" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C235" type2="C293" type3="C135" type4="H140" k1="0" k2="0" k3="-0.158992" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C235" type2="C293" type3="C136" type4="C274" k1="-3.690288" k2="1.4644" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C235" type2="C293" type3="C136" type4="C308" k1="-4.123332" k2="1.61084" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C235" type2="C293" type3="C136" type4="C500" k1="-1.058552" k2="2.0397" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C235" type2="C293" type3="C136" type4="H140" k1="0" k2="0" k3="-0.158992" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C235" type2="C293" type3="C137" type4="H140" k1="0" k2="0" k3="-0.158992" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C235" type2="C293" type3="C149" type4="H140" k1="0" k2="0" k3="-0.158992" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C235" type2="C293" type3="C157" type4="H140" k1="0" k2="0" k3="-0.158992" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C235" type2="C293" type3="C157" type4="O154" k1="-12.118956" k2="0.84726" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C235" type2="C293" type3="C158" type4="H140" k1="0" k2="0" k3="-0.158992" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C235" type2="C293" type3="C158" type4="O154" k1="-12.118956" k2="0.84726" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C235" type2="C293" type3="C206" type4="H140" k1="0" k2="0" k3="-0.158992" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C235" type2="C293" type3="C206" type4="S200" k1="-6.951716" k2="1.106668" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C235" type2="C293" type3="C214" type4="H140" k1="0" k2="0" k3="-0.158992" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C235" type2="C293" type3="C214" type4="S203" k1="-6.951716" k2="1.106668" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C235" type2="C293" type3="C274" type4="C271" k1="3.227956" k2="1.456032" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C235" type2="C293" type3="C274" type4="H140" k1="0" k2="0" k3="-0.158992" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C235" type2="C293" type3="C505" type4="C507" k1="-2.681944" k2="3.44134" k3="-0.035564" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C235" type2="C293" type3="C505" type4="C508" k1="-2.681944" k2="3.44134" k3="-0.035564" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C235" type2="C293" type3="C505" type4="C510" k1="-3.573136" k2="3.171472" k3="-1.050184" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C235" type2="C293" type3="C505" type4="H140" k1="0" k2="0" k3="-0.158992" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C235" type2="C293" type3="N287" type4="H290" k1="0" k2="0" k3="0.725924" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C235" type2="C295" type3="C136" type4="H140" k1="0" k2="0" k3="-0.158992" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C235" type2="C295" type3="N309" type4="C296" k1="3.0080868" k2="-0.2589896" k3="0.5520788" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C235" type2="C295" type3="N309" type4="H310" k1="0" k2="0" k3="0.725924" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C235" type2="N238" type3="C223" type4="H140" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C235" type2="N238" type3="C224" type4="C274" k1="-0.77404" k2="-0.33472" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C235" type2="N238" type3="C224" type4="C505" k1="-0.77404" k2="-0.33472" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C235" type2="N238" type3="C224" type4="H140" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C235" type2="N238" type3="C224I" type4="H140" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C235" type2="N238" type3="C224A" type4="H140" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C235" type2="N238" type3="C224S" type4="H140" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C235" type2="N238" type3="C224K" type4="H140" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C235" type2="N238" type3="C224Y" type4="H140" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C235" type2="N238" type3="C242" type4="H140" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C235" type2="N238" type3="C283" type4="C271" k1="-5.253012" k2="0.43932" k3="-0.4184" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C235" type2="N238" type3="C283" type4="C274" k1="-1.426744" k2="0.27196" k3="0.707096" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C235" type2="N238" type3="C283" type4="C505" k1="-1.426744" k2="0.27196" k3="0.707096" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C235" type2="N238" type3="C283" type4="H140" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C235" type2="N238" type3="C284" type4="C271" k1="-5.253012" k2="0.43932" k3="-0.4184" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C235" type2="N238" type3="C284" type4="H140" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C235" type2="N239" type3="C245" type4="H140" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C235" type2="N239" type3="C246" type4="H140" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C235" type2="N239" type3="C285" type4="C271" k1="-5.253012" k2="0.43932" k3="-0.4184" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C235" type2="N239" type3="C285" type4="H140" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C242" type2="N238" type3="C235" type4="C246" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C242" type2="N238" type3="C235" type4="O236" k1="0" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C245" type2="C136" type3="C136" type4="C246" k1="2.7196" k2="-0.4184" k3="0.4184" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C245" type2="C136" type3="C136" type4="C285" k1="2.7196" k2="-0.4184" k3="0.4184" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C245" type2="C136" type3="C136" type4="H140" k1="0" k2="0" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C245" type2="N239" type3="C235" type4="C246" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C245" type2="N239" type3="C235" type4="C292" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C245" type2="N239" type3="C235" type4="C293" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C245" type2="N239" type3="C235" type4="C295" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C245" type2="N239" type3="C235" type4="O236" k1="0" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C245" type2="N239" type3="C246" type4="H140" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C245" type2="N239" type3="C285" type4="C271" k1="-3.633804" k2="2.617092" k3="-7.324092" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C245" type2="N239" type3="C285" type4="H140" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C246" type2="C136" type3="C136" type4="H140" k1="0" k2="0" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C246" type2="C235" type3="N238" type4="C283" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C246" type2="C235" type3="N238" type4="C284" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C246" type2="C235" type3="N238" type4="H241" k1="0" k2="10.2508" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C246" type2="C235" type3="N239" type4="C246" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C246" type2="C235" type3="N239" type4="C285" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C246" type2="N239" type3="C235" type4="C292" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C246" type2="N239" type3="C235" type4="C293" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C246" type2="N239" type3="C235" type4="C295" k1="4.8116" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C246" type2="N239" type3="C235" type4="O236" k1="0" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C246" type2="N239" type3="C245" type4="H140" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C267" type2="C136" type3="C224" type4="H140" k1="0" k2="0" k3="-0.158992" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C267" type2="C136" type3="C224" type4="N238" k1="-11.508092" k2="3.194484" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C267" type2="C224A" type3="C135" type4="H140" k1="0" k2="0" k3="0.154808" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C267" type2="C223" type3="N238" type4="H241" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C267" type2="C224" type3="N238" type4="H241" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C267" type2="C224I" type3="N238" type4="H241" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C267" type2="C224A" type3="N238" type4="H241" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C267" type2="C224S" type3="N238" type4="H241" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C267" type2="C224K" type3="N238" type4="H241" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C267" type2="C224Y" type3="N238" type4="H241" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C271" type2="C274" type3="C136" type4="C283" k1="-1.85142" k2="2.1443" k3="-2.704956" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C271" type2="C274" type3="C136" type4="C293" k1="-1.85142" k2="2.1443" k3="-2.704956" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C271" type2="C274" type3="C136" type4="H140" k1="0" k2="0" k3="-0.4707" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C271" type2="C274" type3="C224" type4="H140" k1="0" k2="0" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C271" type2="C274" type3="C224" type4="N238" k1="-16.81968" k2="-1.19244" k3="1.52716" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C271" type2="C274" type3="C283" type4="C271" k1="3.227956" k2="1.456032" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C271" type2="C274" type3="C283" type4="H140" k1="0" k2="0" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C271" type2="C274" type3="C283" type4="N238" k1="-16.50588" k2="1.384904" k3="2.085724" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C271" type2="C274" type3="C293" type4="H140" k1="0" k2="0" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C271" type2="C274" type3="C293" type4="N287" k1="-16.50588" k2="1.384904" k3="2.085724" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C271" type2="C283" type3="C135" type4="H140" k1="0" k2="0" k3="-0.158992" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C271" type2="C283" type3="C136" type4="C274" k1="-3.690288" k2="1.4644" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C271" type2="C283" type3="C136" type4="C308" k1="-4.123332" k2="1.61084" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C271" type2="C283" type3="C136" type4="C500" k1="-1.058552" k2="2.0397" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C271" type2="C283" type3="C136" type4="H140" k1="0" k2="0" k3="-0.158992" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C271" type2="C283" type3="C137" type4="H140" k1="0" k2="0" k3="-0.158992" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C271" type2="C283" type3="C149" type4="H140" k1="0" k2="0" k3="-0.158992" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C271" type2="C283" type3="C157" type4="H140" k1="0" k2="0" k3="-0.158992" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C271" type2="C283" type3="C157" type4="O154" k1="-12.118956" k2="0.84726" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C271" type2="C283" type3="C158" type4="H140" k1="0" k2="0" k3="-0.158992" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C271" type2="C283" type3="C158" type4="O154" k1="-12.118956" k2="0.84726" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C271" type2="C283" type3="C206" type4="H140" k1="0" k2="0" k3="-0.158992" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C271" type2="C283" type3="C206" type4="S200" k1="-6.951716" k2="1.106668" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C271" type2="C283" type3="C214" type4="H140" k1="0" k2="0" k3="-0.158992" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C271" type2="C283" type3="C214" type4="S203" k1="-6.951716" k2="1.106668" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C271" type2="C283" type3="C274" type4="H140" k1="0" k2="0" k3="-0.158992" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C271" type2="C283" type3="C505" type4="C507" k1="-2.681944" k2="3.44134" k3="-0.035564" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C271" type2="C283" type3="C505" type4="C508" k1="-2.681944" k2="3.44134" k3="-0.035564" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C271" type2="C283" type3="C505" type4="C510" k1="-3.573136" k2="3.171472" k3="-1.050184" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C271" type2="C283" type3="C505" type4="H140" k1="0" k2="0" k3="-0.158992" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C271" type2="C283" type3="N238" type4="H241" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C271" type2="C284" type3="N238" type4="H241" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C271" type2="C285" type3="C136" type4="H140" k1="0" k2="0" k3="-0.158992" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C274" type2="C136" type3="C224" type4="H140" k1="0" k2="0" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C274" type2="C136" type3="C224" type4="N238" k1="-1.19244" k2="-3.0334" k3="2.78236" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C274" type2="C136" type3="C283" type4="H140" k1="0" k2="0" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C274" type2="C136" type3="C283" type4="N238" k1="4.156804" k2="0.956044" k3="1.71544" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C274" type2="C136" type3="C293" type4="H140" k1="0" k2="0" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C274" type2="C136" type3="C293" type4="N287" k1="4.156804" k2="0.956044" k3="1.71544" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C274" type2="C224" type3="C235" type4="N238" k1="-0.54392" k2="-0.66944" k3="-0.50208" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C274" type2="C224" type3="C235" type4="N239" k1="-0.54392" k2="-0.66944" k3="-0.50208" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C274" type2="C224" type3="C235" type4="O236" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C274" type2="C224" type3="N238" type4="H241" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C274" type2="C283" type3="C271" type4="O272" k1="0" k2="1.71544" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C274" type2="C283" type3="N238" type4="H241" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C274" type2="C293" type3="C235" type4="N238" k1="3.721668" k2="0.876548" k3="-0.23012" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C274" type2="C293" type3="C235" type4="N239" k1="3.721668" k2="0.876548" k3="-0.23012" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C274" type2="C293" type3="C235" type4="O236" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C274" type2="C293" type3="N287" type4="H290" k1="0" k2="0" k3="0.725924" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C283" type2="C136" type3="C136" type4="H140" k1="0" k2="0" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C283" type2="C136" type3="C137" type4="H140" k1="0" k2="0" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C283" type2="C136" type3="C210" type4="H140" k1="0" k2="0" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C283" type2="C136" type3="C210" type4="S202" k1="-3.27398" k2="-0.018828" k3="-0.9414" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C283" type2="C136" type3="C235" type4="N237" k1="3.125448" k2="-1.069012" k3="0.2615" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C283" type2="C136" type3="C235" type4="O236" k1="3.464352" k2="2.727968" k3="0.918388" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C283" type2="C136" type3="C267" type4="O268" k1="2.092" k2="1.142232" k3="0.9414" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C283" type2="C136" type3="C267" type4="O269" k1="0" k2="1.142232" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C283" type2="C136" type3="C274" type4="H140" k1="0" k2="0" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C283" type2="C136" type3="C308" type4="C307" k1="2.7196" k2="-0.4184" k3="0.4184" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C283" type2="C136" type3="C308" type4="H140" k1="0" k2="0" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C283" type2="C136" type3="C500" type4="C501" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C283" type2="C136" type3="C500" type4="C514" k1="-1.493688" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C283" type2="C137" type3="C135" type4="H140" k1="0" k2="0" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C283" type2="C137" type3="C136" type4="H140" k1="0" k2="0" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C283" type2="C157" type3="O154" type4="H155" k1="-0.744752" k2="-0.364008" k3="1.029264" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C283" type2="C158" type3="C135" type4="H140" k1="0" k2="0" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C283" type2="C158" type3="O154" type4="H155" k1="-0.744752" k2="-0.364008" k3="1.029264" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C283" type2="C206" type3="S200" type4="H204" k1="-1.587828" k2="-0.589944" k3="1.42256" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C283" type2="C214" type3="S203" type4="S203" k1="4.060572" k2="-1.748912" k3="1.95602" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C283" type2="C274" type3="C271" type4="O272" k1="5.23" k2="2.092" k3="2.8242" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C283" type2="C505" type3="C507" type4="C508" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C283" type2="C505" type3="C507" type4="N511" k1="-1.17152" k2="-1.54808" k3="0.730108" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C283" type2="C505" type3="C508" type4="C507" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C283" type2="C505" type3="C508" type4="N503" k1="-1.17152" k2="-1.54808" k3="0.730108" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C283" type2="C505" type3="C510" type4="C510" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C283" type2="C505" type3="C510" type4="N512" k1="-8.34708" k2="3.51456" k3="0.60668" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C283" type2="N238" type3="C235" type4="O236" k1="0" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C284" type2="N238" type3="C235" type4="O236" k1="0" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C285" type2="C136" type3="C136" type4="H140" k1="0" k2="0" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C285" type2="N239" type3="C235" type4="O236" k1="0" k2="12.738188" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C285" type2="N239" type3="C245" type4="H140" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C292" type2="C136" type3="C136" type4="H140" k1="0" k2="0" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C292" type2="C235" type3="N238" type4="H241" k1="0" k2="10.2508" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C293" type2="C136" type3="C136" type4="H140" k1="0" k2="0" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C293" type2="C136" type3="C137" type4="H140" k1="0" k2="0" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C293" type2="C136" type3="C210" type4="H140" k1="0" k2="0" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C293" type2="C136" type3="C210" type4="S202" k1="-3.27398" k2="-0.018828" k3="-0.9414" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C293" type2="C136" type3="C235" type4="N237" k1="3.125448" k2="-1.069012" k3="0.2615" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C293" type2="C136" type3="C235" type4="O236" k1="3.464352" k2="2.727968" k3="0.918388" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C293" type2="C136" type3="C267" type4="O268" k1="2.092" k2="1.142232" k3="0.9414" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C293" type2="C136" type3="C267" type4="O269" k1="0" k2="1.142232" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C293" type2="C136" type3="C274" type4="H140" k1="0" k2="0" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C293" type2="C136" type3="C308" type4="C307" k1="2.7196" k2="-0.4184" k3="0.4184" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C293" type2="C136" type3="C308" type4="H140" k1="0" k2="0" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C293" type2="C136" type3="C500" type4="C501" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C293" type2="C136" type3="C500" type4="C514" k1="-1.493688" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C293" type2="C137" type3="C135" type4="H140" k1="0" k2="0" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C293" type2="C137" type3="C136" type4="H140" k1="0" k2="0" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C293" type2="C157" type3="O154" type4="H155" k1="-0.744752" k2="-0.364008" k3="1.029264" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C293" type2="C158" type3="C135" type4="H140" k1="0" k2="0" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C293" type2="C158" type3="O154" type4="H155" k1="-0.744752" k2="-0.364008" k3="1.029264" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C293" type2="C206" type3="S200" type4="H204" k1="-1.587828" k2="-0.589944" k3="1.42256" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C293" type2="C214" type3="S203" type4="S203" k1="4.060572" k2="-1.748912" k3="1.95602" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C293" type2="C235" type3="N238" type4="H241" k1="0" k2="10.2508" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C293" type2="C274" type3="C271" type4="O272" k1="5.23" k2="2.092" k3="2.8242" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C293" type2="C505" type3="C507" type4="C508" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C293" type2="C505" type3="C507" type4="N511" k1="-1.17152" k2="-1.54808" k3="0.730108" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C293" type2="C505" type3="C508" type4="C507" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C293" type2="C505" type3="C508" type4="N503" k1="-1.17152" k2="-1.54808" k3="0.730108" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C293" type2="C505" type3="C510" type4="C510" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C293" type2="C505" type3="C510" type4="N512" k1="-8.34708" k2="3.51456" k3="0.60668" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C295" type2="C136" type3="C136" type4="C296" k1="2.7196" k2="-0.4184" k3="0.4184" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C295" type2="C136" type3="C136" type4="H140" k1="0" k2="0" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C295" type2="C235" type3="N238" type4="H241" k1="0" k2="10.2508" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C295" type2="N309" type3="C296" type4="H140" k1="0" k2="0" k3="0.6311564" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C296" type2="C136" type3="C136" type4="H140" k1="0" k2="0" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C296" type2="N309" type3="C295" type4="H140" k1="0" k2="0" k3="0.6311564" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C302" type2="N303" type3="C307" type4="C308" k1="3.826268" k2="0.508356" k3="-1.041816" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C302" type2="N303" type3="C307" type4="H140" k1="0" k2="0" k3="0.370284" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C307" type2="C308" type3="C136" type4="H140" k1="0" k2="0" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C307" type2="N303" type3="C302" type4="N300" k1="0" k2="16.602112" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C308" type2="C136" type3="C224" type4="H140" k1="0" k2="0" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C308" type2="C136" type3="C224" type4="N238" k1="1.99786" k2="-0.30334" k3="0.58576" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C308" type2="C136" type3="C283" type4="H140" k1="0" k2="0" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C308" type2="C136" type3="C283" type4="N238" k1="0.215476" k2="1.366076" k3="1.177796" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C308" type2="C136" type3="C293" type4="H140" k1="0" k2="0" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C308" type2="C136" type3="C293" type4="N287" k1="0.215476" k2="1.366076" k3="1.177796" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C308" type2="C307" type3="N303" type4="H304" k1="-0.39748" k2="-0.872364" k3="0.874456" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C500" type2="C136" type3="C224" type4="H140" k1="0" k2="0" k3="0.966504" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C500" type2="C136" type3="C224" type4="N238" k1="2.28028" k2="1.23428" k3="1.84096" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C500" type2="C136" type3="C283" type4="H140" k1="0" k2="0" k3="0.966504" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C500" type2="C136" type3="C283" type4="N238" k1="-1.230096" k2="2.13384" k3="1.39118" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C500" type2="C136" type3="C293" type4="H140" k1="0" k2="0" k3="0.966504" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C500" type2="C136" type3="C293" type4="N287" k1="-1.230096" k2="2.13384" k3="1.39118" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C500" type2="C501" type3="C145" type4="H146" k1="0" k2="14.644" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C501" type2="C500" type3="C136" type4="H140" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C501" type2="C502" type3="C145" type4="H146" k1="0" k2="15.167" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C502" type2="C501" type3="C145" type4="H146" k1="0" k2="14.644" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C502" type2="C501" type3="C500" type4="C514" k1="0" k2="7.0082" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C502" type2="N503" type3="C514" type4="H146" k1="0" k2="6.276" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C505" type2="C224" type3="C235" type4="N238" k1="-0.54392" k2="-0.66944" k3="-0.50208" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C505" type2="C224" type3="C235" type4="N239" k1="-0.54392" k2="-0.66944" k3="-0.50208" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C505" type2="C224" type3="C235" type4="O236" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C505" type2="C224" type3="N238" type4="H241" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C505" type2="C283" type3="C271" type4="O272" k1="0" k2="1.71544" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C505" type2="C283" type3="N238" type4="H241" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C505" type2="C293" type3="C235" type4="N238" k1="3.721668" k2="0.876548" k3="-0.23012" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C505" type2="C293" type3="C235" type4="O236" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C505" type2="C293" type3="N287" type4="H290" k1="0" k2="0" k3="0.725924" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C505" type2="C508" type3="C507" type4="H146" k1="0" k2="22.489" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C505" type2="C508" type3="C507" type4="N511" k1="0" k2="22.489" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C505" type2="C508" type3="N503" type4="C506" k1="0" k2="5.8576" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C505" type2="C508" type3="N503" type4="H504" k1="0" k2="5.8576" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C505" type2="C510" type3="N512" type4="C509" k1="0" k2="5.8576" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C505" type2="C510" type3="N512" type4="H513" k1="0" k2="5.8576" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C506" type2="N503" type3="C508" type4="C507" k1="0" k2="5.8576" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C506" type2="N503" type3="C508" type4="H146" k1="0" k2="10.46" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C506" type2="N511" type3="C507" type4="C508" k1="0" k2="10.0416" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C506" type2="N511" type3="C507" type4="H146" k1="0" k2="10.0416" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C507" type2="C505" type3="C224" type4="H140" k1="0" k2="0" k3="0.966504" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C507" type2="C505" type3="C224" type4="N238" k1="-6.52704" k2="0.81588" k3="1.92464" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C507" type2="C505" type3="C283" type4="H140" k1="0" k2="0" k3="0.966504" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C507" type2="C505" type3="C283" type4="N238" k1="-1.133864" k2="0.91002" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C507" type2="C505" type3="C293" type4="H140" k1="0" k2="0" k3="0.966504" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C507" type2="C505" type3="C293" type4="N287" k1="-1.133864" k2="0.91002" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C507" type2="C508" type3="C505" type4="H140" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C507" type2="C508" type3="N503" type4="H504" k1="0" k2="5.8576" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C507" type2="N511" type3="C506" type4="H146" k1="0" k2="20.92" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C507" type2="N511" type3="C506" type4="N503" k1="0" k2="20.92" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C508" type2="C505" type3="C224" type4="H140" k1="0" k2="0" k3="0.966504" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C508" type2="C505" type3="C224" type4="N238" k1="-1.4644" k2="0.39748" k3="1.046" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C508" type2="C505" type3="C283" type4="H140" k1="0" k2="0" k3="0.966504" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C508" type2="C505" type3="C283" type4="N238" k1="-1.133864" k2="0.91002" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C508" type2="C505" type3="C293" type4="H140" k1="0" k2="0" k3="0.966504" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C508" type2="C505" type3="C293" type4="N287" k1="-1.133864" k2="0.91002" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C508" type2="C507" type3="C505" type4="H140" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C508" type2="N503" type3="C506" type4="H146" k1="0" k2="9.7278" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C508" type2="N503" type3="C506" type4="N511" k1="0" k2="9.7278" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C509" type2="N512" type3="C510" type4="C510" k1="0" k2="5.8576" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C509" type2="N512" type3="C510" type4="H146" k1="0" k2="10.46" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C510" type2="C505" type3="C224" type4="H140" k1="0" k2="0" k3="0.966504" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C510" type2="C505" type3="C224" type4="N238" k1="-2.74052" k2="-0.77404" k3="2.05016" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C510" type2="C505" type3="C283" type4="H140" k1="0" k2="0" k3="0.966504" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C510" type2="C505" type3="C283" type4="N238" k1="-6.355496" k2="0.876548" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C510" type2="C505" type3="C293" type4="H140" k1="0" k2="0" k3="0.966504" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C510" type2="C505" type3="C293" type4="N287" k1="-6.355496" k2="0.876548" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C510" type2="C510" type3="C505" type4="H140" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C510" type2="C510" type3="N512" type4="H513" k1="0" k2="5.8576" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C510" type2="N512" type3="C509" type4="H146" k1="0" k2="9.7278" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C510" type2="N512" type3="C509" type4="N512" k1="0" k2="9.7278" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C514" type2="C500" type3="C136" type4="H140" k1="0" k2="0" k3="-1.00416" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C135" type3="C136" type4="H140" k1="0" k2="0" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C135" type3="C137" type4="H140" k1="0" k2="0" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C135" type3="C158" type4="H140" k1="0" k2="0" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C135" type3="C158" type4="O154" k1="0" k2="0" k3="0.979056" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C135" type3="C224A" type4="H140" k1="0" k2="0" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C135" type3="C224A" type4="N238" k1="0" k2="0" k3="0.970688" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C135" type3="C235" type4="N238" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C135" type3="C235" type4="N239" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C135" type3="C235" type4="O236" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C135" type3="C283" type4="H140" k1="0" k2="0" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C135" type3="C283" type4="N238" k1="0" k2="0" k3="0.970688" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C135" type3="C293" type4="H140" k1="0" k2="0" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C135" type3="C293" type4="N287" k1="0" k2="0" k3="0.970688" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C136" type3="C136" type4="H140" k1="0" k2="0" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C136" type3="C137" type4="H140" k1="0" k2="0" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C136" type3="C210" type4="H140" k1="0" k2="0" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C136" type3="C210" type4="S202" k1="0" k2="0" k3="0.945584" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C136" type3="C224" type4="H140" k1="0" k2="0" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C136" type3="C224K" type4="H140" k1="0" k2="0" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C136" type3="C224" type4="N238" k1="0" k2="0" k3="0.970688" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C136" type3="C224K" type4="N238" k1="0" k2="0" k3="0.970688" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C136" type3="C235" type4="N237" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C136" type3="C235" type4="O236" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C136" type3="C245" type4="H140" k1="0" k2="0" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C136" type3="C245" type4="N239" k1="0" k2="0" k3="0.970688" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C136" type3="C246" type4="H140" k1="0" k2="0" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C136" type3="C246" type4="N239" k1="0" k2="0" k3="0.970688" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C136" type3="C267" type4="O268" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C136" type3="C267" type4="O269" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C136" type3="C274" type4="H140" k1="0" k2="0" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C136" type3="C283" type4="H140" k1="0" k2="0" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C136" type3="C283" type4="N238" k1="0" k2="0" k3="0.970688" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C136" type3="C285" type4="H140" k1="0" k2="0" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C136" type3="C285" type4="N239" k1="0" k2="0" k3="0.970688" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C136" type3="C292" type4="H140" k1="0" k2="0" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C136" type3="C292" type4="N287" k1="0" k2="0" k3="0.803328" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C136" type3="C293" type4="H140" k1="0" k2="0" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C136" type3="C293" type4="N287" k1="0" k2="0" k3="0.970688" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C136" type3="C295" type4="H140" k1="0" k2="0" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C136" type3="C295" type4="N309" k1="0" k2="0" k3="0.803328" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C136" type3="C296" type4="H140" k1="0" k2="0" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C136" type3="C296" type4="N309" k1="0" k2="0" k3="0.803328" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C136" type3="C308" type4="H140" k1="0" k2="0" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C137" type3="C224" type4="H140" k1="0" k2="0" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C137" type3="C224I" type4="H140" k1="0" k2="0" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C137" type3="C224" type4="N238" k1="0" k2="0" k3="0.970688" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C137" type3="C224I" type4="N238" k1="0" k2="0" k3="0.970688" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C137" type3="C283" type4="H140" k1="0" k2="0" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C137" type3="C283" type4="N238" k1="0" k2="0" k3="0.970688" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C137" type3="C293" type4="H140" k1="0" k2="0" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C137" type3="C293" type4="N287" k1="0" k2="0" k3="0.970688" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C149" type3="C224" type4="H140" k1="0" k2="0" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C149" type3="C224Y" type4="H140" k1="0" k2="0" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C149" type3="C224" type4="N238" k1="0" k2="0" k3="0.970688" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C149" type3="C224Y" type4="N238" k1="0" k2="0" k3="0.970688" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C149" type3="C283" type4="H140" k1="0" k2="0" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C149" type3="C283" type4="N238" k1="0" k2="0" k3="0.970688" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C149" type3="C293" type4="H140" k1="0" k2="0" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C149" type3="C293" type4="N287" k1="0" k2="0" k3="0.970688" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C157" type3="C224S" type4="H140" k1="0" k2="0" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C157" type3="C224S" type4="N238" k1="0" k2="0" k3="0.970688" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C157" type3="C283" type4="H140" k1="0" k2="0" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C157" type3="C283" type4="N238" k1="0" k2="0" k3="0.970688" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C157" type3="C293" type4="H140" k1="0" k2="0" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C157" type3="C293" type4="N287" k1="0" k2="0" k3="0.970688" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C157" type3="O154" type4="H155" k1="0" k2="0" k3="0.7372208" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C158" type3="C224S" type4="H140" k1="0" k2="0" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C158" type3="C224S" type4="N238" k1="0" k2="0" k3="0.970688" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C158" type3="C283" type4="H140" k1="0" k2="0" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C158" type3="C283" type4="N238" k1="0" k2="0" k3="0.970688" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C158" type3="C293" type4="H140" k1="0" k2="0" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C158" type3="C293" type4="N287" k1="0" k2="0" k3="0.970688" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C158" type3="O154" type4="H155" k1="0" k2="0" k3="0.7372208" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C206" type3="C224" type4="H140" k1="0" k2="0" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C206" type3="C224" type4="N238" k1="0" k2="0" k3="0.970688" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C206" type3="C283" type4="H140" k1="0" k2="0" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C206" type3="C283" type4="N238" k1="0" k2="0" k3="0.970688" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C206" type3="C293" type4="H140" k1="0" k2="0" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C206" type3="C293" type4="N287" k1="0" k2="0" k3="0.970688" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C206" type3="S200" type4="H204" k1="1.94556" k2="6.0668" k3="-0.66944" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C214" type3="C224" type4="H140" k1="0" k2="0" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C214" type3="C224" type4="N238" k1="0" k2="0" k3="0.970688" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C214" type3="C283" type4="H140" k1="0" k2="0" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C214" type3="C283" type4="N238" k1="0" k2="0" k3="0.970688" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C214" type3="C293" type4="H140" k1="0" k2="0" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C214" type3="C293" type4="N287" k1="0" k2="0" k3="0.970688" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C214" type3="S203" type4="S203" k1="0" k2="0" k3="1.167336" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C223" type3="C235" type4="N238" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C223" type3="C235" type4="N239" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C223" type3="C235" type4="O236" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C223" type3="N238" type4="H241" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C224S" type3="C157" type4="O154" k1="0" k2="0" k3="0.979056" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C224S" type3="C158" type4="O154" k1="0" k2="0" k3="0.979056" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C224" type3="C206" type4="S200" k1="0" k2="0" k3="0.945584" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C224" type3="C214" type4="S203" k1="0" k2="0" k3="0.945584" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C224" type3="C235" type4="N238" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C224I" type3="C235" type4="N238" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C224A" type3="C235" type4="N238" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C224S" type3="C235" type4="N238" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C224K" type3="C235" type4="N238" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C224Y" type3="C235" type4="N238" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C224" type3="C235" type4="N239" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C224I" type3="C235" type4="N239" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C224A" type3="C235" type4="N239" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C224S" type3="C235" type4="N239" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C224K" type3="C235" type4="N239" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C224Y" type3="C235" type4="N239" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C224" type3="C235" type4="O236" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C224I" type3="C235" type4="O236" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C224A" type3="C235" type4="O236" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C224S" type3="C235" type4="O236" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C224K" type3="C235" type4="O236" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C224Y" type3="C235" type4="O236" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C223" type3="C267" type4="O268" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C223" type3="C267" type4="O269" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C224" type3="C267" type4="O268" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C224I" type3="C267" type4="O268" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C224A" type3="C267" type4="O268" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C224S" type3="C267" type4="O268" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C224K" type3="C267" type4="O268" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C224Y" type3="C267" type4="O268" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C224" type3="C267" type4="O269" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C224I" type3="C267" type4="O269" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C224A" type3="C267" type4="O269" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C224S" type3="C267" type4="O269" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C224K" type3="C267" type4="O269" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C224Y" type3="C267" type4="O269" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C224" type3="C274" type4="H140" k1="0" k2="0" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C224" type3="C505" type4="H140" k1="0" k2="0" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C224" type3="N238" type4="H241" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C224I" type3="N238" type4="H241" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C224A" type3="N238" type4="H241" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C224S" type3="N238" type4="H241" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C224K" type3="N238" type4="H241" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C224Y" type3="N238" type4="H241" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C242" type3="N238" type4="H241" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C246" type3="C235" type4="N238" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C246" type3="C235" type4="N239" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C246" type3="C235" type4="O236" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C274" type3="C224" type4="N238" k1="0" k2="0" k3="0.970688" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C274" type3="C271" type4="O272" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C274" type3="C283" type4="H140" k1="0" k2="0" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C274" type3="C283" type4="N238" k1="0" k2="0" k3="0.970688" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C274" type3="C293" type4="H140" k1="0" k2="0" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C274" type3="C293" type4="N287" k1="0" k2="0" k3="0.970688" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C283" type3="C157" type4="O154" k1="0" k2="0" k3="0.979056" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C283" type3="C158" type4="O154" k1="0" k2="0" k3="0.979056" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C283" type3="C206" type4="S200" k1="0" k2="0" k3="0.945584" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C283" type3="C214" type4="S203" k1="0" k2="0" k3="0.945584" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C283" type3="C271" type4="O272" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C283" type3="C505" type4="H140" k1="0" k2="0" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C283" type3="N238" type4="H241" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C284" type3="C271" type4="O272" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C284" type3="N238" type4="H241" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C285" type3="C271" type4="O272" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C292" type3="C235" type4="N238" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C292" type3="C235" type4="O236" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C292" type3="N287" type4="H290" k1="0" k2="0" k3="0.546012" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C293" type3="C157" type4="O154" k1="0" k2="0" k3="0.979056" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C293" type3="C158" type4="O154" k1="0" k2="0" k3="0.979056" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C293" type3="C206" type4="S200" k1="0" k2="0" k3="0.945584" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C293" type3="C214" type4="S203" k1="0" k2="0" k3="0.945584" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C293" type3="C235" type4="N238" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C293" type3="C235" type4="N239" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C293" type3="C235" type4="O236" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C293" type3="C505" type4="H140" k1="0" k2="0" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C293" type3="N287" type4="H290" k1="0" k2="0" k3="0.546012" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C295" type3="C235" type4="N238" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C295" type3="C235" type4="N239" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C295" type3="C235" type4="O236" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C295" type3="N309" type4="H310" k1="0" k2="0" k3="0.546012" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C296" type3="N309" type4="H310" k1="0" k2="0" k3="0.803328" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C307" type3="C308" type4="H140" k1="0" k2="0" k3="0.6276" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C307" type3="N303" type4="H304" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C308" type3="C307" type4="N303" k1="0" k2="0" k3="-1.217544" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C505" type3="C224" type4="N238" k1="0" k2="0" k3="0.970688" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C505" type3="C283" type4="N238" k1="0" k2="0" k3="0.970688" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C505" type3="C293" type4="N287" k1="0" k2="0" k3="0.970688" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C505" type3="C507" type4="N511" k1="0" k2="0" k3="0.876548" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C505" type3="C508" type4="N503" k1="0" k2="0" k3="0.876548" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H140" type2="C505" type3="C510" type4="N512" k1="0" k2="0" k3="0.876548" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H146" type2="C508" type3="C507" type4="N511" k1="0" k2="22.489" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H146" type2="C508" type3="N503" type4="H504" k1="0" k2="10.46" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H146" type2="C510" type3="N512" type4="H513" k1="0" k2="10.46" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H240" type2="N237" type3="C235" type4="O236" k1="0" k2="10.2508" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H241" type2="N238" type3="C235" type4="O236" k1="0" k2="10.2508" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H270" type2="O268" type3="C267" type4="O269" k1="0" k2="11.506" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H301" type2="N300" type3="C302" type4="N300" k1="0" k2="8.1588" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H301" type2="N300" type3="C302" type4="N303" k1="0" k2="8.1588" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H304" type2="N303" type3="C302" type4="N300" k1="0" k2="8.1588" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H504" type2="N503" type3="C506" type4="N511" k1="0" k2="9.7278" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="H513" type2="N512" type3="C509" type4="N512" k1="0" k2="9.7278" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="N238" type2="C223" type3="C235" type4="N238" k1="0.29288" k2="1.08784" k3="-0.69036" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="N238" type2="C223" type3="C235" type4="N239" k1="0.29288" k2="1.08784" k3="-0.69036" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="N238" type2="C223" type3="C235" type4="O236" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="N238" type2="C224S" type3="C157" type4="O154" k1="11.12944" k2="1.569" k3="0.48116" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="N238" type2="C224S" type3="C158" type4="O154" k1="7.61488" k2="5.94128" k3="8.17972" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="N238" type2="C224" type3="C206" type4="S200" k1="6.92452" k2="0.60668" k3="3.59824" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="N238" type2="C224" type3="C214" type4="S203" k1="4.29906" k2="1.106668" k3="1.138048" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="N238" type2="C224" type3="C235" type4="N238" k1="0.18828" k2="0.89956" k3="-0.60668" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="N238" type2="C224I" type3="C235" type4="N238" k1="0.18828" k2="0.89956" k3="-0.60668" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="N238" type2="C224K" type3="C235" type4="N238" k1="0.18828" k2="0.89956" k3="-0.60668" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="N238" type2="C224S" type3="C235" type4="N238" k1="-0.06276" k2="0" k3="0.25104" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="N238" type2="C224Y" type3="C235" type4="N238" k1="0.18828" k2="0.89956" k3="-0.60668" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="N238" type2="C224A" type3="C235" type4="N238" k1="0.18828" k2="0.87864" k3="-0.60668" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="N238" type2="C224A" type3="C235" type4="N239" k1="0.37656" k2="1.75728" k3="-1.21336" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="N238" type2="C224S" type3="C235" type4="N239" k1="-0.12552" k2="0" k3="0.50208" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="N238" type2="C224K" type3="C235" type4="N239" k1="0.37656" k2="1.79912" k3="-1.21336" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="N238" type2="C224Y" type3="C235" type4="N239" k1="0.37656" k2="1.79912" k3="-1.21336" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="N238" type2="C224I" type3="C235" type4="N239" k1="0.18828" k2="0.89956" k3="-0.60668" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="N238" type2="C224" type3="C235" type4="N239" k1="0.18828" k2="0.89956" k3="-0.60668" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="N238" type2="C224I" type3="C235" type4="O236" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="N238" type2="C224" type3="C235" type4="O236" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="N238" type2="C224A" type3="C235" type4="O236" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="N238" type2="C224S" type3="C235" type4="O236" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="N238" type2="C224K" type3="C235" type4="O236" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="N238" type2="C224Y" type3="C235" type4="O236" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="N238" type2="C223" type3="C267" type4="O268" k1="11.00392" k2="1.71544" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="N238" type2="C224" type3="C267" type4="O268" k1="11.00392" k2="1.71544" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="N238" type2="C224I" type3="C267" type4="O268" k1="11.00392" k2="1.71544" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="N238" type2="C224A" type3="C267" type4="O268" k1="11.00392" k2="1.71544" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="N238" type2="C224S" type3="C267" type4="O268" k1="11.00392" k2="1.71544" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="N238" type2="C224K" type3="C267" type4="O268" k1="11.00392" k2="1.71544" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="N238" type2="C224Y" type3="C267" type4="O268" k1="11.00392" k2="1.71544" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="N238" type2="C223" type3="C267" type4="O269" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="N238" type2="C224" type3="C267" type4="O269" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="N238" type2="C224I" type3="C267" type4="O269" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="N238" type2="C224A" type3="C267" type4="O269" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="N238" type2="C224S" type3="C267" type4="O269" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="N238" type2="C224K" type3="C267" type4="O269" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="N238" type2="C224Y" type3="C267" type4="O269" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="N238" type2="C235" type3="C246" type4="N239" k1="4.41412" k2="1.9874" k3="-4.93712" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="N238" type2="C235" type3="C292" type4="N287" k1="3.78652" k2="4.50826" k3="-0.98324" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="N238" type2="C235" type3="C293" type4="N287" k1="3.78652" k2="4.50826" k3="-0.98324" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="N238" type2="C235" type3="C295" type4="N309" k1="-1.96648" k2="5.76346" k3="-5.58564" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="N238" type2="C283" type3="C157" type4="O154" k1="13.091736" k2="-2.169404" k3="2.859764" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="N238" type2="C283" type3="C158" type4="O154" k1="13.091736" k2="-2.169404" k3="2.859764" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="N238" type2="C283" type3="C206" type4="S200" k1="4.29906" k2="1.106668" k3="1.138048" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="N238" type2="C283" type3="C214" type4="S203" k1="4.29906" k2="1.106668" k3="1.138048" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="N238" type2="C283" type3="C271" type4="O272" k1="0" k2="1.71544" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="N238" type2="C284" type3="C271" type4="O272" k1="0" k2="1.71544" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="N239" type2="C235" type3="C246" type4="N239" k1="4.41412" k2="1.9874" k3="-4.93712" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="N239" type2="C235" type3="C293" type4="N287" k1="3.78652" k2="4.50826" k3="-0.98324" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="N239" type2="C235" type3="C295" type4="N309" k1="-1.96648" k2="5.76346" k3="-5.58564" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="N239" type2="C246" type3="C235" type4="O236" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="N239" type2="C285" type3="C271" type4="O272" k1="0" k2="1.71544" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="N287" type2="C292" type3="C235" type4="O236" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="N287" type2="C293" type3="C157" type4="O154" k1="13.091736" k2="-2.169404" k3="2.859764" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="N287" type2="C293" type3="C158" type4="O154" k1="13.091736" k2="-2.169404" k3="2.859764" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="N287" type2="C293" type3="C206" type4="S200" k1="4.29906" k2="1.106668" k3="1.138048" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="N287" type2="C293" type3="C214" type4="S203" k1="4.29906" k2="1.106668" k3="1.138048" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="N287" type2="C293" type3="C235" type4="O236" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="N309" type2="C295" type3="C235" type4="O236" k1="0" k2="0" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="N503" type2="C508" type3="C507" type4="N511" k1="0" k2="22.489" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C267" type2="C136" type3="C293" type4="N287" k1="-11.508092" k2="3.194484" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C235" type2="C293" type3="C136" type4="C267" k1="1.251016" k2="3.259336" k3="0.53346" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C267" type2="C136" type3="C283" type4="N238" k1="-11.508092" k2="3.194484" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="C267" type2="C136" type3="C283" type4="C271" k1="1.251016" k2="3.259336" k3="0.53346" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="" type2="C145" type3="C145" type4="" k1="0" k2="15.167" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="" type2="C145" type3="C166" type4="" k1="0" k2="15.167" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="" type2="C166" type3="C145" type4="" k1="0" k2="15.167" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="" type2="C145" type3="C501" type4="" k1="0" k2="14.644" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="" type2="C501" type3="C145" type4="" k1="0" k2="14.644" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="" type2="C145" type3="C502" type4="" k1="0" k2="15.167" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="" type2="C502" type3="C145" type4="" k1="0" k2="15.167" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="" type2="N300" type3="C302" type4="" k1="0" k2="8.1588" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="" type2="C302" type3="N300" type4="" k1="0" k2="8.1588" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="" type2="C500" type3="C501" type4="" k1="0" k2="7.0082" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="" type2="C501" type3="C500" type4="" k1="0" k2="7.0082" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="" type2="C500" type3="C514" type4="" k1="0" k2="27.3006" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="" type2="C514" type3="C500" type4="" k1="0" k2="27.3006" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="" type2="C501" type3="C502" type4="" k1="0" k2="12.552" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="" type2="C502" type3="C501" type4="" k1="0" k2="12.552" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="" type2="N503" type3="C502" type4="" k1="0" k2="6.3806" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="" type2="C502" type3="N503" type4="" k1="0" k2="6.3806" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="" type2="C506" type3="N503" type4="" k1="0" k2="9.7278" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="" type2="N503" type3="C506" type4="" k1="0" k2="9.7278" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="" type2="N511" type3="C506" type4="" k1="0" k2="20.92" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="" type2="C506" type3="N511" type4="" k1="0" k2="20.92" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="" type2="C507" type3="C508" type4="" k1="0" k2="22.489" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="" type2="C508" type3="C507" type4="" k1="0" k2="22.489" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="" type2="N511" type3="C507" type4="" k1="0" k2="10.0416" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="" type2="C507" type3="N511" type4="" k1="0" k2="10.0416" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="" type2="N512" type3="C509" type4="" k1="0" k2="9.7278" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="" type2="C509" type3="N512" type4="" k1="0" k2="9.7278" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="" type2="C510" type3="C510" type4="" k1="0" k2="22.489" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="" type2="N503" type3="C514" type4="" k1="0" k2="6.276" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Proper type1="" type2="C514" type3="N503" type4="" k1="0" k2="6.276" k3="0" k4="0.0" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00" phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
    <Improper type1="C500" type2="C136" type3="C501" type4="C514" k1="4.6024" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="C500" type2="C136" type3="C514" type4="C501" k1="4.6024" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="C502" type2="C145" type3="C501" type4="N503" k1="4.6024" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="C502" type2="C145" type3="N503" type4="C501" k1="4.6024" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="C145" type2="C149" type3="C145" type4="C145" k1="4.6024" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="N238" type2="C245" type3="C235" type4="C285" k1="10.46" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="N238" type2="C245" type3="C285" type4="C235" k1="10.46" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="N239" type2="C245" type3="C235" type4="C246" k1="10.46" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="N239" type2="C245" type3="C235" type4="C285" k1="10.46" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="N239" type2="C245" type3="C246" type4="C235" k1="10.46" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="N239" type2="C245" type3="C285" type4="C235" k1="10.46" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="C501" type2="C502" type3="C145" type4="C500" k1="4.6024" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="C501" type2="C502" type3="C500" type4="C145" k1="4.6024" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="C507" type2="C505" type3="C508" type4="N511" k1="4.6024" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="C507" type2="C505" type3="N511" type4="C508" k1="4.6024" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="C508" type2="C505" type3="C507" type4="N503" k1="4.6024" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="C508" type2="C505" type3="N503" type4="C507" k1="4.6024" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="C510" type2="C505" type3="C510" type4="N512" k1="4.6024" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="C510" type2="C505" type3="N512" type4="C510" k1="4.6024" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="C145" type2="H146" type3="C145" type4="C145" k1="4.6024" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="C145" type2="H146" type3="C166" type4="C145" k1="4.6024" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="C145" type2="H146" type3="C145" type4="C166" k1="4.6024" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="C145" type2="H146" type3="C145" type4="C501" k1="4.6024" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="C145" type2="H146" type3="C501" type4="C145" k1="4.6024" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="C145" type2="H146" type3="C145" type4="C502" k1="4.6024" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="C145" type2="H146" type3="C502" type4="C145" k1="4.6024" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="C506" type2="H146" type3="N503" type4="N511" k1="4.6024" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="C506" type2="H146" type3="N511" type4="N503" k1="4.6024" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="C507" type2="H146" type3="C508" type4="N511" k1="4.6024" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="C507" type2="H146" type3="N511" type4="C508" k1="4.6024" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="C508" type2="H146" type3="C507" type4="N503" k1="4.6024" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="C508" type2="H146" type3="N503" type4="C507" k1="4.6024" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="C509" type2="H146" type3="N512" type4="N512" k1="4.6024" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="C510" type2="H146" type3="C510" type4="N512" k1="4.6024" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="C510" type2="H146" type3="N512" type4="C510" k1="4.6024" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="C514" type2="H146" type3="C500" type4="N503" k1="4.6024" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="C514" type2="H146" type3="N503" type4="C500" k1="4.6024" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="N237" type2="H240" type3="C235" type4="H240" k1="10.46" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="N237" type2="H240" type3="H240" type4="C235" k1="10.46" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="N238" type2="H241" type3="C223" type4="C235" k1="10.46" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="N238" type2="H241" type3="C224" type4="C235" k1="10.46" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="N238" type2="H241" type3="C224I" type4="C235" k1="10.46" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="N238" type2="H241" type3="C224A" type4="C235" k1="10.46" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="N238" type2="H241" type3="C224S" type4="C235" k1="10.46" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="N238" type2="H241" type3="C224K" type4="C235" k1="10.46" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="N238" type2="H241" type3="C224Y" type4="C235" k1="10.46" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="N238" type2="H241" type3="C235" type4="C223" k1="10.46" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="N238" type2="H241" type3="C235" type4="C224" k1="10.46" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="N238" type2="H241" type3="C235" type4="C224I" k1="10.46" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="N238" type2="H241" type3="C235" type4="C224A" k1="10.46" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="N238" type2="H241" type3="C235" type4="C224S" k1="10.46" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="N238" type2="H241" type3="C235" type4="C224K" k1="10.46" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="N238" type2="H241" type3="C235" type4="C224Y" k1="10.46" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="N238" type2="H241" type3="C235" type4="C242" k1="10.46" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="N238" type2="H241" type3="C235" type4="C283" k1="10.46" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="N238" type2="H241" type3="C235" type4="C284" k1="10.46" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="N238" type2="H241" type3="C242" type4="C235" k1="10.46" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="N238" type2="H241" type3="C283" type4="C235" k1="10.46" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="N238" type2="H241" type3="C284" type4="C235" k1="10.46" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="N300" type2="H301" type3="C302" type4="H301" k1="10.46" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="N300" type2="H301" type3="H301" type4="C302" k1="10.46" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="N303" type2="H304" type3="C302" type4="C307" k1="10.46" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="N303" type2="H304" type3="C307" type4="C302" k1="10.46" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="N503" type2="H504" type3="C502" type4="C514" k1="10.46" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="N503" type2="H504" type3="C506" type4="C508" k1="10.46" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="N503" type2="H504" type3="C508" type4="C506" k1="10.46" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="N503" type2="H504" type3="C514" type4="C502" k1="10.46" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="N512" type2="H513" type3="C509" type4="C510" k1="10.46" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="N512" type2="H513" type3="C510" type4="C509" k1="10.46" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="C302" type2="N300" type3="N300" type4="N303" k1="43.932" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="C302" type2="N300" type3="N303" type4="N300" k1="43.932" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="C166" type2="O167" type3="C145" type4="C145" k1="4.6024" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="C235" type2="O236" type3="C135" type4="N239" k1="43.932" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="C235" type2="O236" type3="C135" type4="N238" k1="43.932" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="C235" type2="O236" type3="C136" type4="N237" k1="43.932" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="C235" type2="O236" type3="C223" type4="N238" k1="43.932" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="C235" type2="O236" type3="C223" type4="N239" k1="43.932" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="C235" type2="O236" type3="C224" type4="N238" k1="43.932" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="C235" type2="O236" type3="C224I" type4="N238" k1="43.932" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="C235" type2="O236" type3="C224A" type4="N238" k1="43.932" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="C235" type2="O236" type3="C224S" type4="N238" k1="43.932" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="C235" type2="O236" type3="C224K" type4="N238" k1="43.932" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="C235" type2="O236" type3="C224Y" type4="N238" k1="43.932" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="C235" type2="O236" type3="C224" type4="N239" k1="43.932" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="C235" type2="O236" type3="C224I" type4="N239" k1="43.932" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="C235" type2="O236" type3="C224A" type4="N239" k1="43.932" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="C235" type2="O236" type3="C224S" type4="N239" k1="43.932" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="C235" type2="O236" type3="C224K" type4="N239" k1="43.932" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="C235" type2="O236" type3="C224Y" type4="N239" k1="43.932" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="C235" type2="O236" type3="C246" type4="N238" k1="43.932" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="C235" type2="O236" type3="C246" type4="N239" k1="43.932" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="C235" type2="O236" type3="C292" type4="N238" k1="43.932" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="C235" type2="O236" type3="C293" type4="N238" k1="43.932" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="C235" type2="O236" type3="C293" type4="N239" k1="43.932" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="C235" type2="O236" type3="C295" type4="N238" k1="43.932" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="C235" type2="O236" type3="C295" type4="N239" k1="43.932" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="C235" type2="O236" type3="N237" type4="C136" k1="43.932" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="C235" type2="O236" type3="N238" type4="C135" k1="43.932" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="C235" type2="O236" type3="N238" type4="C223" k1="43.932" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="C235" type2="O236" type3="N238" type4="C224" k1="43.932" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="C235" type2="O236" type3="N238" type4="C224I" k1="43.932" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="C235" type2="O236" type3="N238" type4="C224A" k1="43.932" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="C235" type2="O236" type3="N238" type4="C224S" k1="43.932" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="C235" type2="O236" type3="N238" type4="C224K" k1="43.932" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="C235" type2="O236" type3="N238" type4="C224Y" k1="43.932" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="C235" type2="O236" type3="N238" type4="C246" k1="43.932" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="C235" type2="O236" type3="N238" type4="C292" k1="43.932" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="C235" type2="O236" type3="N238" type4="C293" k1="43.932" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="C235" type2="O236" type3="N238" type4="C295" k1="43.932" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="C235" type2="O236" type3="N239" type4="C135" k1="43.932" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="C235" type2="O236" type3="N239" type4="C223" k1="43.932" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="C235" type2="O236" type3="N239" type4="C224" k1="43.932" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="C235" type2="O236" type3="N239" type4="C224A" k1="43.932" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="C235" type2="O236" type3="N239" type4="C224S" k1="43.932" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="C235" type2="O236" type3="N239" type4="C224K" k1="43.932" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="C235" type2="O236" type3="N239" type4="C224Y" k1="43.932" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="C235" type2="O236" type3="N239" type4="C246" k1="43.932" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="C235" type2="O236" type3="N239" type4="C293" k1="43.932" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="C235" type2="O236" type3="N239" type4="C295" k1="43.932" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="C267" type2="O269" type3="C136" type4="O268" k1="43.932" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="C267" type2="O269" type3="O268" type4="C136" k1="43.932" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="C267" type2="O269" type3="C224" type4="O268" k1="43.932" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="C267" type2="O269" type3="C224I" type4="O268" k1="43.932" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="C267" type2="O269" type3="C224A" type4="O268" k1="43.932" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="C267" type2="O269" type3="C224S" type4="O268" k1="43.932" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="C267" type2="O269" type3="C224K" type4="O268" k1="43.932" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="C267" type2="O269" type3="C224Y" type4="O268" k1="43.932" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="C267" type2="O269" type3="O268" type4="C224" k1="43.932" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="C267" type2="O269" type3="O268" type4="C224I" k1="43.932" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="C267" type2="O269" type3="O268" type4="C224A" k1="43.932" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="C267" type2="O269" type3="O268" type4="C224S" k1="43.932" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="C267" type2="O269" type3="O268" type4="C224K" k1="43.932" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="C267" type2="O269" type3="O268" type4="C224Y" k1="43.932" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="C271" type2="O272" type3="C274" type4="O272" k1="43.932" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="C271" type2="O272" type3="C283" type4="O272" k1="43.932" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="C271" type2="O272" type3="C284" type4="O272" k1="43.932" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="C271" type2="O272" type3="C285" type4="O272" k1="43.932" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="C271" type2="O272" type3="O272" type4="C274" k1="43.932" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="C271" type2="O272" type3="O272" type4="C283" k1="43.932" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="C271" type2="O272" type3="O272" type4="C284" k1="43.932" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="C271" type2="O272" type3="O272" type4="C285" k1="43.932" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="C267" type2="O268" type3="C136" type4="O269" k1="43.932" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="C267" type2="O268" type3="C223" type4="O269" k1="43.932" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="C267" type2="O268" type3="C224" type4="O269" k1="43.932" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="C267" type2="O268" type3="C224I" type4="O269" k1="43.932" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="C267" type2="O268" type3="C224A" type4="O269" k1="43.932" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="C267" type2="O268" type3="C224S" type4="O269" k1="43.932" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="C267" type2="O268" type3="C224K" type4="O269" k1="43.932" periodicity1="2" phase1="3.141592653589793" />
    <Improper type1="C267" type2="O268" type3="C224Y" type4="O269" k1="43.932" periodicity1="2" phase1="3.141592653589793" />
  </PeriodicTorsionForce>
  <NonbondedForce coulomb14scale="0.5" lj14scale="0.5">
    <UseAttributeFromResidue name="charge"/>
    <Atom epsilon="1" sigma="1" type="C235"/>
    <Atom epsilon="1" sigma="1" type="C267"/>
    <Atom epsilon="1" sigma="1" type="C271"/>
    <Atom epsilon="1" sigma="1" type="C145"/>
    <Atom epsilon="1" sigma="1" type="C166"/>
    <Atom epsilon="1" sigma="1" type="C302"/>
    <Atom epsilon="1" sigma="1" type="C501"/>
    <Atom epsilon="1" sigma="1" type="C502"/>
    <Atom epsilon="1" sigma="1" type="C506"/>
    <Atom epsilon="1" sigma="1" type="C509"/>
    <Atom epsilon="1" sigma="1" type="C500"/>
    <Atom epsilon="1" sigma="1" type="C135"/>
    <Atom epsilon="1" sigma="1" type="C136"/>
    <Atom epsilon="1" sigma="1" type="C137"/>
    <Atom epsilon="1" sigma="1" type="C149"/>
    <Atom epsilon="1" sigma="1" type="C157"/>
    <Atom epsilon="1" sigma="1" type="C158"/>
    <Atom epsilon="1" sigma="1" type="C206"/>
    <Atom epsilon="1" sigma="1" type="C209"/>
    <Atom epsilon="1" sigma="1" type="C210"/>
    <Atom epsilon="1" sigma="1" type="C214"/>
    <Atom epsilon="1" sigma="1" type="C223"/>
    <Atom epsilon="1" sigma="1" type="C224"/>
    <Atom epsilon="1" sigma="1" type="C224A"/>
    <Atom epsilon="1" sigma="1" type="C224I"/>
    <Atom epsilon="1" sigma="1" type="C224S"/>
    <Atom epsilon="1" sigma="1" type="C224Y"/>
    <Atom epsilon="1" sigma="1" type="C224K"/>
    <Atom epsilon="1" sigma="1" type="C242"/>
    <Atom epsilon="1" sigma="1" type="C245"/>
    <Atom epsilon="1" sigma="1" type="C246"/>
    <Atom epsilon="1" sigma="1" type="C274"/>
    <Atom epsilon="1" sigma="1" type="C283"/>
    <Atom epsilon="1" sigma="1" type="C284"/>
    <Atom epsilon="1" sigma="1" type="C285"/>
    <Atom epsilon="1" sigma="1" type="C292"/>
    <Atom epsilon="1" sigma="1" type="C293"/>
    <Atom epsilon="1" sigma="1" type="C295"/>
    <Atom epsilon="1" sigma="1" type="C296"/>
    <Atom epsilon="1" sigma="1" type="C307"/>
    <Atom epsilon="1" sigma="1" type="C308"/>
    <Atom epsilon="1" sigma="1" type="C505"/>
    <Atom epsilon="1" sigma="1" type="C507"/>
    <Atom epsilon="1" sigma="1" type="C508"/>
    <Atom epsilon="1" sigma="1" type="C514"/>
    <Atom epsilon="1" sigma="1" type="C510"/>
    <Atom epsilon="1" sigma="1" type="H240"/>
    <Atom epsilon="1" sigma="1" type="H241"/>
    <Atom epsilon="1" sigma="1" type="H504"/>
    <Atom epsilon="1" sigma="1" type="H513"/>
    <Atom epsilon="1" sigma="1" type="H290"/>
    <Atom epsilon="1" sigma="1" type="H301"/>
    <Atom epsilon="1" sigma="1" type="H304"/>
    <Atom epsilon="1" sigma="1" type="H310"/>
    <Atom epsilon="1" sigma="1" type="H146"/>
    <Atom epsilon="1" sigma="1" type="H140"/>
    <Atom epsilon="1" sigma="1" type="H155"/>
    <Atom epsilon="1" sigma="1" type="H168"/>
    <Atom epsilon="1" sigma="1" type="H270"/>
    <Atom epsilon="1" sigma="1" type="H204"/>
    <Atom epsilon="1" sigma="1" type="N237"/>
    <Atom epsilon="1" sigma="1" type="N238"/>
    <Atom epsilon="1" sigma="1" type="N239"/>
    <Atom epsilon="1" sigma="1" type="N300"/>
    <Atom epsilon="1" sigma="1" type="N303"/>
    <Atom epsilon="1" sigma="1" type="N287"/>
    <Atom epsilon="1" sigma="1" type="N309"/>
    <Atom epsilon="1" sigma="1" type="N503"/>
    <Atom epsilon="1" sigma="1" type="N512"/>
    <Atom epsilon="1" sigma="1" type="N511"/>
    <Atom epsilon="1" sigma="1" type="O236"/>
    <Atom epsilon="1" sigma="1" type="O269"/>
    <Atom epsilon="1" sigma="1" type="O272"/>
    <Atom epsilon="1" sigma="1" type="O154"/>
    <Atom epsilon="1" sigma="1" type="O167"/>
    <Atom epsilon="1" sigma="1" type="O268"/>
    <Atom epsilon="1" sigma="1" type="S202"/>
    <Atom epsilon="1" sigma="1" type="S203"/>
    <Atom epsilon="1" sigma="1" type="S200"/>
    <!--WATER -->
    <Atom epsilon="0.635968" sigma="0.31507524065751241" type="tip3p-O"/>
    <Atom epsilon="0" sigma="1" type="tip3p-H"/>
    <!--DATA FOR IONS    -->
    <Atom epsilon="0" sigma="1" type="SOD"/>
    <Atom epsilon="0" sigma="1" type="CLA"/>
  </NonbondedForce>
</ForceField>''')