#Bond and angle related functions used to optimize molecules in Psi4

#get all of the global config settings
def config_setup():
    import configparser, os
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

#convert the ligpargen pdb to psi4 (may need different parser for other pdbs?)
def pdb_to_psi4(inputfile, charge, multiplicity, basis, theory):
    import re, os
    molecule_name=inputfile[:-4]
    with open(inputfile, 'r') as pdb:
         lines=pdb.readlines()
         molecule=[]
         for line in lines:
             if 'ATOM' in line:
                 #Find the element type from PDB
                 element = str(line.split()[2])[:-1]
                 element = re.sub('[0-9]+', '', element)
                 molecule.append([element, line.split()[5], line.split()[6], line.split()[7]])
    out = open(molecule_name+'.psi4in', 'w+')
    out.write('''molecule %s {
 %s %s \n'''%(molecule_name, charge, multiplicity))
    for i in range(len(molecule)):
        out.write(' %s    %9.6f  %9.6f  %9.6f\n'%(str(molecule[i][0]), float(molecule[i][1]) ,float(molecule[i][2]) , float(molecule[i][3])))
    out.write('''}

set basis %s
gradient('%s')'''%(basis, theory.lower()))
    try:
       os.mkdir('BONDS')
    except:
       pass
    os.system('mv %s.psi4in BONDS'%(molecule_name))
 









