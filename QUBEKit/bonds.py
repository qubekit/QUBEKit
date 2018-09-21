#Bond and angle related functions used to optimize molecules in Psi4
import configparser, os, sys
#get all of the global config settings
def config_setup():
    config = configparser.RawConfigParser()
    #look for config file 
    loc=os.environ.get("QUBEKit")
    try: 
        with open(os.path.join(loc,"QUBEKit/new_config.ini")) as source:
            config.readfp( source )
            theory = str(config['qm']['theory'])
    except:
        with open(os.path.join(loc,"QUBEKit/config.ini")) as source:
            config.readfp( source )
            theory = str(config['qm']['theory'])
    #config.read('config.ini')
    basis = str(config['qm']['basis'])
    vib_scaling= float(config['qm']['vib_scaling'])
    processors= int(config['qm']['processors'])
    memory= int(config['qm']['memory'])
    dihstart= int(config['fitting']['dihstart'])
    increment= int(config['fitting']['increment'])
    numscan= int(config['fitting']['numscan'])
    T_weight= str(config['fitting']['T_weight'])
    new_dihnum= int(config['fitting']['new_dihnum'])
    Q_file= config['fitting']['Q_file']
    tor_limit= int(config['fitting']['tor_limit'])
    div_index = int(config['fitting']['div_index'])
    return theory, basis, vib_scaling, processors, memory, dihstart, increment, numscan, T_weight, new_dihnum, Q_file, tor_limit, div_index



#read molecule pdb file can read ATOM or HETATM formats with only one molecule per pdb
def read_pdb(inputfile):
    import re
    with open(inputfile,'r') as pdb:
         lines=pdb.readlines()
         molecule=[]
         for line in lines:
             if 'ATOM' in line:
                element = str(line[76:78])
                #if the eleement coloum is missing from the pdb extract the element from the name
                if len(element) < 1:
                   element = str(line.split()[2])[:-1]
                   element = re.sub('[0-9]+', '', element)
                molecule.append([element, float(line[30:38]), float(line[38:46]), float(line[46:54]) ])
             elif 'HETATM' in line:
                element = str(line[76:78]) 
                #if the eleement coloum is missing from the pdb extract the element from the name
                if len(element) < 1:
                   element = str(line.split()[2])[:-1]
                   element = re.sub('[0-9]+', '', element)
                molecule.append([element, float(line[30:38]), float(line[38:46]), float(line[46:54]) ])   
    return molecule

#convert the pdb data to psi4in format ready for geometric 
def pdb_to_psi4_geo(inputfile, molecule, charge, multiplicity, basis, theory):
    import os
    molecule_name=inputfile[:-4]
    out = open(molecule_name+'.psi4in', 'w+')
    out.write('''molecule %s {
 %s %s \n'''%(molecule_name, charge, multiplicity))
    for i in range(len(molecule)):
        out.write(' %s    %9.6f  %9.6f  %9.6f\n'%(str(molecule[i][0]), float(molecule[i][1]) ,float(molecule[i][2]) , float(molecule[i][3])))
    out.write('''}

set basis %s
gradient('%s')'''%(basis, theory.lower()))
    out.close()
    try:
       os.mkdir('BONDS')
    except:
       pass
    os.system('mv %s.psi4in BONDS'%(molecule_name))
    view=input('''psi4 input made and moved to BONDS
would you like to view the file
>''') 
    if view.lower() == 'y' or view.lower() == 'yes':
        psi4in = open('BONDS/%s.psi4in'%(molecule_name),'r').read()
        print(psi4in)


#convert to psi4 format to be run in psi4 without geometric 
def pdb_to_psi4(inputfile, molecule, charge, multiplicity, basis, theory, memory):
    import os
    molecule_name=inputfile[:-4]
    out = open('input.dat', 'w+')
    out.write('''memory %s GB

molecule %s {
 %s %s \n'''%(memory, molecule_name, charge, multiplicity))
    for i in range(len(molecule)):
        out.write(' %s    %9.6f  %9.6f  %9.6f\n'%(str(molecule[i][0]), float(molecule[i][1]) ,float(molecule[i][2]) , float(molecule[i][3])))
    out.write('''}

set basis %s
optimize('%s')
frequency('%s')
set hessian_write on
hessian('%s')
'''%(basis, theory.lower(), theory.lower(), theory.lower()))
    out.close()
    try:
       os.mkdir('BONDS')
    except:
       pass
    os.system('mv input.dat BONDS')
    view=input('''psi4 input made and moved to BONDS
would you like to view the file
>''') 
    if view.lower() == 'y' or view.lower() == 'yes':
        psi4in = open('BONDS/input.dat','r').read()
        print(psi4in)


#write the psi4 frequency and hessian input file
def freq_psi4(inputfile, opt_moleucle, charge, multiplicity, basis, theory, memory):
    molecule_name=inputfile[:-4]
    try:
       os.mkdir('FREQ')
    except:
       pass
    os.chdir('FREQ')
    out = open('%s_freq.dat'%(molecule_name),'w+')
    out.write('''memory %s GB

molecule %s {
 %s %s \n'''%(memory, molecule_name, charge, multiplicity))
    for i in range(len(opt_moleucle)):
        out.write(' %s    %9.6f  %9.6f  %9.6f\n'%(str(opt_moleucle[i][0]), float(opt_moleucle[i][1]) ,float(opt_moleucle[i][2]) , float(opt_moleucle[i][3])))
    out.write('''}

set basis %s
frequency('%s')
set hessian_write on
hessian('%s')
'''%(basis, theory.lower(), theory.lower()))
    out.close()


#function to use the modified seminario method
def modified_seminario():
    pass

#write g09 optimization and freq input files 
def pdb_to_g09(inputfile, molecule, charge, multiplicity, basis, theory, processors, memory):
    molecule_name=inputfile[:-4]
    out=open('%s.com'%(molecule_name),'w+')
    out.write("""%%Mem=%sGB
%%NProcShared=%s
%%Chk=lig
# %s/%s SCF=XQC Opt=tight freq
    
ligand
    
%s %s
"""%(memory, processors, theory, basis, charge, multiplicity))
    for i in range(len(molecule)):
       out.write('%-2s      %9.6f    %9.6f    %9.6f\n'%(str(molecule[i][0]), float(molecule[i][1]) ,float(molecule[i][2]) , float(molecule[i][3])))
    out.write('\n')
    out.write('\n')
    out.write('\n')
    out.write('\n')
    out.close()
    try:
       os.mkdir('BONDS')
    except:
       pass
    os.system('mv %s.com BONDS'%(molecule_name))
    view=input('''gaussian09 input made and moved to BONDS
would you like to view the file
>''') 
    if view.lower() == 'y' or view.lower() == 'yes':
        g09 = open('BONDS/%s.com'%(molecule_name),'r').read()
        print(g09)

#function to get the hessian matrix from psi4
def extract_hessian_psi4(molecule):
    import glob
    import numpy as np
    try:
       for file in glob.glob("*.hess"):
           hessian=file
           print('Hessian found')
       hessian
    except:        
         sys.exit('Hessian missing!')
    hess_raw=[]   
    with open(hessian,'r') as hess:
         lines = hess.readlines() 
         for line in lines[1:]:
             for i in range(3):
                 hess_raw.append(float(line.split()[i]))
    hess_raw = np.array(hess_raw)
    hess_form = np.reshape(hess_raw, (3*len(molecule), 3*len(molecule)))* (627.509391) / (0.529**2)  #Change from Hartree/bohr to kcal/mol /ang
    return hess_form
   
       


#get the optimized structure from psi4 ready for the frequency
def get_molecule_from_psi4():
    opt_molecule=[]
    write=False
    with open('opt.xyz','r') as opt:
            lines = opt.readlines()
            for line in lines:
                if 'Iteration' in line:
                    print('optimization converged at iteration %s with final energy %s'%(int(line.split()[1]), float(line.split()[3])))
                    write=True
                elif write:
                   opt_molecule.append([str(line.split()[0]), float(line.split()[1]), float(line.split()[2]), float(line.split()[3])])    
    return opt_molecule

#function to get the optimized molecule from g09 
def get_molecule_from_g09():
    pass



