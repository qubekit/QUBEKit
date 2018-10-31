#TODO josh finish getting the seminario method to work from any qm engine hessian should be in np.array format

def modified_Seminario_method(vibrational_scaling, molecule):
    """Calculate the new bond and angle terms after being passed the symetric hessian and optimized
     molecule may also need the a parameter file"""
    #   Program to implement the Modified Seminario Method
    #   Written by Alice E. A. Allen, TCM, University of Cambridge
    #   Modified by Joshua T. Horton University of Newcastle 
    #   Reference using AEA Allen, MC Payne, DJ Cole, J. Chem. Theory Comput. (2018), doi:10.1021/acs.jctc.7b00785

    # Pass as arguements the input folder containing the zmat.log/lig.log,
    # lig.fchk and optional Zmat.z file, the output folder where the new
    # parameters will be written and the vibrational frequency scaling constant
    # required.

    import time
    import numpy as np
    import os.path 

    #Create log file
    # TODO this needs changing to match the other log files
    fid_log = open('MSM_log', "w+")
    fid_log.write('Modified Seminario Method \n')
    fid_log.write('Parametrization started\n')
    fid_log.write('Time is now: '+ time.strftime('%X %x %Z') + '\n')

    # Square the vibrational scaling used for frequencies
    vibrational_scaling_squared = vibrational_scaling**2; 

    # Import all input data from gaussian outputfile
    # TODO once new class system works these steps can be removed!
    # if engine == 'g09':
    #   [ bond_list, angle_list, coords, N, hessian, atom_names ] = input_data_processing_g09()
    #elif engine == 'psi4':
    N = int(len(molecule.QMoptimized)) # only want to deal with the qm structure here.
    #numpy object of the hessian
    hessian = molecule.hessian  # the molecule brings its own hessian with it.
    coords = []
    for i in range(len(opt_molecule)):
        for j in range(3):
             coords.append(opt_molecule[i][j+1])
    coords = np.reshape(coords ,(N,3))
    print(coords)
       # atom names will just be atom numbers
       atom_names = []
       # bond and angle list should come from the pdb from the intial paramertization?
       #[ bond_list, angle_list ] = input_data_processing_psi4()
    #Find bond lengths
    bond_lengths = np.zeros((N, N))

    for i in range (N):
        for j in range(N):
            diff_i_j = np.array(coords[i,:]) - np.array(coords[j,:])
            bond_lengths[i][j] =  np.linalg.norm(diff_i_j)
    print(bond_lengths)

    eigenvectors = np.empty((3, 3, N, N), dtype=complex)
    eigenvalues = np.empty((N, N, 3), dtype=complex)
    partial_hessian = np.zeros((3, 3))

    for i in range(0,N):
        for j in range(0,N):
            partial_hessian = hessian[(i * 3):((i + 1)*3),(j * 3):((j + 1)*3)]
            [a, b] = np.linalg.eig(partial_hessian)
            eigenvalues[i,j,:] = (a)
            eigenvectors[:,:,i,j] = (b)

    # The bond values are calculated and written to file
    unique_values_bonds = bonds_calculated_printed( outputfilefolder, vibrational_scaling_squared, bond_list, bond_lengths, atom_names, eigenvalues, eigenvectors, coords )

    # The angle values are calculated and written to file
    unique_values_angles = angles_calculated_printed( outputfilefolder, vibrational_scaling_squared, angle_list, bond_lengths, atom_names, eigenvalues, eigenvectors, coords )

    #The final section finds the average bond and angle terms for each
    #bond/angle class if the .z exists to supply angle/bond classes and then 
    #writes the new terms to a .sb file
    if os.path.exists(inputfilefolder + 'Zmat.z'):
        average_values_across_classes( unique_values_bonds, unique_values_angles, outputfilefolder )
        sb_file_new_parameters(outputfilefolder, 'Python_Modified_Scaled')
        
def angles_calculated_printed( outputfilefolder, vibrational_scaling_squared, angle_list, bond_lengths, atom_names, eigenvalues, eigenvectors, coords ):
    #This function uses the modified Seminario method to find the angle
    #parameters and print them to file

    from operator import itemgetter
    import numpy as np
    from force_angle_constant import u_PA_from_angles, force_angle_constant

    #Open output file angle parameters are written to 
    fid = open(outputfilefolder + 'Modified_Seminario_Angle', 'w')

    k_theta = np.zeros(len(angle_list))
    theta_0 = np.zeros(len(angle_list))
    unique_values_angles = [] #Used to find average values

    #Modified Seminario part goes here ...
    #Connectivity information for Modified Seminario Method
    central_atoms_angles = []

    #A structure is created with the index giving the central atom of the
    #angle, an array then lists the angles with that central atom. 
    #ie. central_atoms_angles{3} contains an array of angles with central atom
    #3
    for i in range(0, len(coords)):
        central_atoms_angles.append([])
        for j in range(0, len(angle_list)):
            if i == angle_list[j][1]:
                #For angle ABC, atoms A C are written to array
                AC_array = [angle_list[j][0],  angle_list[j][2], j]
                central_atoms_angles[i].append(AC_array)

                #For angle ABC, atoms C A are written to array
                CA_array = [angle_list[j][2],  angle_list[j][0], j]
                central_atoms_angles[i].append(CA_array)

    #Sort rows by atom number
    for i in range(0, len(coords)):
        central_atoms_angles[i] = sorted(central_atoms_angles[i], key=itemgetter(0))

    #Find normals u_PA for each angle
    unit_PA_all_angles = []

    for i in range(0,len(central_atoms_angles)):
        unit_PA_all_angles.append([])
        for j in range(0, len(central_atoms_angles[i])):
        #For the angle at central_atoms_angles[i][j,:] the corresponding
        #u_PA value is found for the plane ABC and bond AB, where ABC
        #corresponds to the order of the arguements
        #This is why the reverse order was also added
            unit_PA_all_angles[i].append(u_PA_from_angles(central_atoms_angles[i][j][0], i, central_atoms_angles[i][j][1], coords))

    #Finds the contributing factors from the other angle terms
    #scaling_factor_all_angles = cell(max(max(angle_list))); %This array will contain scaling factor and angle list position
    scaling_factor_all_angles = []

    for i in range(0,len(central_atoms_angles)):
        scaling_factor_all_angles.append([])
        for j in range(0,len(central_atoms_angles[i])):
            n = 1
            m = 1
            angles_around = 0 
            additional_contributions = 0 
            scaling_factor_all_angles[i].append([0,0]) 
        
            #Position in angle list
            scaling_factor_all_angles[i][j][1] =  central_atoms_angles[i][j][2] 
        
            #Goes through the list of angles with the same central atom
            #And computes the term need for the modified Seminario method    

            #Forwards directions, finds the same bonds with the central atom i  
            while( ( (j + n ) < len(central_atoms_angles[i]) ) and central_atoms_angles[i][j][0] == central_atoms_angles[i][j+n][0] ):
                additional_contributions = additional_contributions + (abs(np.dot(unit_PA_all_angles[i][j][:], unit_PA_all_angles[i][j + n][:])))**2 
                n = n + 1
                angles_around = angles_around + 1
        
            #Backwards direction, finds the same bonds with the central atom i   
            while( ( (j - m ) >= 0 ) and central_atoms_angles[i][j][0] == central_atoms_angles[i][j-m][0] ):
                additional_contributions = additional_contributions + (abs(np.dot(unit_PA_all_angles[i][j][:], unit_PA_all_angles[i][j - m][:] ) ) )**2
                m = m + 1
                angles_around =  angles_around + 1
        
            if (n != 1 or m != 1):
                #Finds the mean value of the additional contribution
                #To change to normal Seminario method comment out + part 
                scaling_factor_all_angles[i][j][0] = 1 + ( additional_contributions / (m  + n - 2) )  
            else:
                scaling_factor_all_angles[i][j][0] = 1

    scaling_factors_angles_list = []
    for i in range(0,len(angle_list) ):
        scaling_factors_angles_list.append([]) 

    #Orders the scaling factors according to the angle list
    for i in range(0,len(central_atoms_angles)):
        for j in range(0,len(central_atoms_angles[i]) ):
            scaling_factors_angles_list[scaling_factor_all_angles[i][j][1]].append(scaling_factor_all_angles[i][j][0]) 

    #Finds the angle force constants with the scaling factors included for each angle
    for i in range(0,len(angle_list) ):
        #Ensures that there is no difference when the ordering is changed 
        [AB_k_theta, AB_theta_0] = force_angle_constant( angle_list[i][0], angle_list[i][1], angle_list[i][2], bond_lengths, eigenvalues, eigenvectors, coords, scaling_factors_angles_list[i][0], scaling_factors_angles_list[i][1] ) 
        [BA_k_theta, BA_theta_0] = force_angle_constant( angle_list[i][2], angle_list[i][1], angle_list[i][0], bond_lengths, eigenvalues, eigenvectors, coords, scaling_factors_angles_list[i][1], scaling_factors_angles_list[i][0] ) 
        k_theta[i] = ( AB_k_theta + BA_k_theta ) / 2
        theta_0[i] = ( AB_theta_0 +  BA_theta_0 ) / 2
    
        #Vibrational_scaling takes into account DFT deficities/ anharmocity 
        k_theta[i] =  k_theta[i] * vibrational_scaling_squared
        
        fid.write(str(i))
        fid.write( '  ' + atom_names[angle_list[i][0]] + '-' + atom_names[angle_list[i][1] ] + '-' + atom_names[angle_list[i][2]] + '  ' )
        fid.write(str("%.3f" % k_theta[i]) + '   ' + str("%.3f" % theta_0[i]) + '   ' + str(angle_list[i][0] + 1)  + '   ' + str(angle_list[i][1] + 1) + '   ' + str(angle_list[i][2] + 1))
        fid.write('\n')

        unique_values_angles.append([atom_names[angle_list[i][0]], atom_names[angle_list[i][1]], atom_names[angle_list[i][2]], k_theta[i], theta_0[i], 1 ])

    fid.close

    return unique_values_angles

def u_PA_from_angles( atom_A, atom_B, atom_C, coords ):
    #This gives the vector in the plane A,B,C and perpendicular to A to B
    from unit_vector_N import unit_vector_N
    import numpy as np 

    diff_AB = coords[:,atom_B] - coords[:, atom_A]
    norm_diff_AB = np.linalg.norm(diff_AB)
    u_AB = diff_AB / norm_diff_AB 

    diff_CB = coords[:,atom_B] - coords[:, atom_C]
    norm_diff_CB = np.linalg.norm(diff_CB)
    u_CB = diff_CB / norm_diff_CB 

    u_N = unit_vector_N( u_CB, u_AB )

    u_PA = cross(u_N,  u_AB)
    norm_diff_PA = np.linalg.norm(diff_PA)
    u_PA = u_PA /norm_u_PA

    return u_PA

def average_values_across_classes( unique_values_bonds, unique_values_angles, outputfilefolder ):
    #This function finds the average bond and angle term for each class. 
    
    ignore_rows = [] 

    #Find Average Values Bonds
    for i in range(0,(len(unique_values_bonds))):
        for j in range((i+1),len(unique_values_bonds)):
            #Finds if the bond class has already been encountered
            if ( unique_values_bonds[i][0] == unique_values_bonds[j][0] ) and (unique_values_bonds[i][1] ==  unique_values_bonds[j][1] )  or ( ( unique_values_bonds[i][0] == unique_values_bonds[j][1] ) and (unique_values_bonds[i][1] == unique_values_bonds[j][0] ) ):
                unique_values_bonds[i][2] = unique_values_bonds[i][2] + unique_values_bonds[j][2]
                unique_values_bonds[i][3] = unique_values_bonds[i][3] + unique_values_bonds[j][3]
                unique_values_bonds[i][4] = unique_values_bonds[i][4] + 1
                ignore_rows.append(j)
    
    #Average Bonds Printed
    fid = open(( outputfilefolder + 'Average_Modified_Seminario_Bonds' ), 'w')

    #Remove bond classes that were already present and find mean value
    for i in range(0,len(unique_values_bonds)):
        if (i in ignore_rows) == 0:
            unique_values_bonds[i][2] = unique_values_bonds[i][2] / unique_values_bonds[i][4]
            unique_values_bonds[i][3] = unique_values_bonds[i][3] / unique_values_bonds[i][4]
            fid.write( unique_values_bonds[i][0] + '-' + unique_values_bonds[i][1] + '  ' + str("%.2f" % unique_values_bonds[i][2]) + '  ' + str("%.3f" % unique_values_bonds[i][3]) + '\n' )

    fid.close()

    #Find Average Values Angles
    ignore_rows_angles = []
    
    #Find Average Values Angles
    for i in range(0,(len(unique_values_angles))):
        for j in range((i+1),len(unique_values_angles)):
            #Finds if the angle class has already been encountered
            if ( unique_values_angles[i][0] ==  unique_values_angles[j][0] and unique_values_angles[i][1] ==  unique_values_angles[j][1] and unique_values_angles[i][2] ==  unique_values_angles[j][2] ) or ( unique_values_angles[i][0] == unique_values_angles[j][2]  and unique_values_angles[i][1] == unique_values_angles[j][1] and unique_values_angles[i][2] == unique_values_angles[j][0] ):
                unique_values_angles[i][3] = unique_values_angles[i][3] + unique_values_angles[j][3]
                unique_values_angles[i][4] = unique_values_angles[i][4] + unique_values_angles[j][4]
                unique_values_angles[i][5] = unique_values_angles[i][5] + 1
                ignore_rows_angles.append(j)

    #Average Angles Printed
    fid = open(( outputfilefolder + 'Average_Modified_Seminario_Angles' ), 'w')

    #Remove angles classes that were already present and find mean value
    for i in range(0,len(unique_values_angles)):
        if (i in ignore_rows_angles ) == 0:
            unique_values_angles[i][3] = unique_values_angles[i][3] / unique_values_angles[i][5]
            unique_values_angles[i][4] = unique_values_angles[i][4] / unique_values_angles[i][5]
            fid.write( unique_values_angles[i][0] + '-' + unique_values_angles[i][1] + '-' + unique_values_angles[i][2] + '  ' + str("%.2f" % unique_values_angles[i][3]) + '  ' + str("%.3f" % unique_values_angles[i][4]) + '\n' )

    fid.close()

def bonds_calculated_printed(outputfilefolder, vibrational_scaling_squared, bond_list, bond_lengths, atom_names, eigenvalues, eigenvectors, coords):
    #This function uses the Seminario method to find the bond
    #parameters and print them to file

    import numpy as np
    from force_constant_bond import force_constant_bond

    #Open output file bond parameters are written to 
    fid = open((outputfilefolder + 'Modified_Seminario_Bonds'), 'w')
    
    k_b = np.zeros(len(bond_list))
    bond_length_list = np.zeros(len(bond_list))
    unique_values_bonds = [] # Used to find average values 

    for i in range(0, len(bond_list)):
        AB = force_constant_bond(bond_list[i][0], bond_list[i][1],eigenvalues, eigenvectors, coords)
        BA = force_constant_bond(bond_list[i][1], bond_list[i][0],eigenvalues, eigenvectors, coords)
        
        # Order of bonds sometimes causes slight differences, find the mean
        k_b[i] = np.real(( AB + BA ) /2); 

        # Vibrational_scaling takes into account DFT deficities/ anharmocity    
        k_b[i] = k_b[i] * vibrational_scaling_squared

        bond_length_list[i] =  bond_lengths[bond_list[i][0]][bond_list[i][1]]
        fid.write(atom_names[bond_list[i][0]] + '-' + atom_names[bond_list[i][1]] + '  ')
        fid.write(str("%.3f" % k_b[i])+ '   ' + str("%.3f" % bond_length_list[i]) +  '   ' +
                  str(bond_list[i][0] + 1) +  '   ' + str(bond_list[i][1] + 1))
        fid.write('\n')

        unique_values_bonds.append([atom_names[bond_list[i][0]], atom_names[bond_list[i][1]], k_b[i], bond_length_list[i], 1 ])
    
    fid.close()

    return unique_values_bonds

def coords_from_fchk(inputfilefolder, fchk_file):
    #Function extracts xyz file from the .fchk output file from Gaussian, this
    #provides the coordinates of the molecules
    import os.path
    import numpy as np

    if os.path.exists(inputfilefolder + fchk_file):
        fid = open((inputfilefolder + fchk_file), "r")
    else:
        fid_log = open((inputfilefolder + 'MSM_log'), "a")
        fid_log.write('ERROR = No .fchk file found.')
        fid_log.close()
        return 0,0
        
    tline = fid.readline()

    loop = 'y'
    numbers = [] #Atomic numbers for use in xyz file
    list_coords = [] #List of xyz coordinates
    hessian = []

    #Get atomic number and coordinates from fchk 
    while tline:
        #Atomic Numbers found
        if len(tline) > 16 and (tline[0:15].strip() == 'Atomic numbers'):
            tline = fid.readline()
            while len(tline) < 17 or (tline[0:16].strip() != 'Nuclear charges'):
                tmp = (tline.strip()).split()
                numbers.extend(tmp)
                tline = fid.readline()
            
        #Get coordinates
        if len(tline) > 31 and tline[0:31].strip() == 'Current cartesian coordinates':
            tline = fid.readline()
            while len(tline) < 15 or ( tline[0:14].strip() != 'Force Field' and tline[0:17].strip() != 'Int Atom Types'and tline[0:13].strip() != 'Atom Types'):
                tmp = (tline.strip()).split()
                list_coords.extend(tmp)
                tline = fid.readline()

        #Gets Hessian 
        if len(tline) > 25 and (tline[0:24].strip() == 'Cartesian Force Constants'):
            tline = fid.readline()
            while len(tline) < 13 or (tline[0:12].strip() != 'Nuclear charges'):
                tmp = (tline.strip()).split()
                np.append(hessian, tmp, 0)
                tline = fid.readline()

            loop = 'n'

        tline = fid.readline()

    fid.close()

    list_coords = [float(x)*float(0.529) for x in list_coords]

    N = int( float(len(list_coords)) / 3.0 ) #Number of atoms

    #Opens the new xyz file 
    file = open(inputfilefolder + 'input_coords.xyz', "w")
    file.write(str(N) + '\n \n')

    xyz = np.zeros((N,3))
    n = 0

    names = []

    fid_csv = open('elementlist.csv', "r")

    with fid_csv as f:
        lines = fid_csv.read().splitlines()

    element_names = []

    #Turn list in a matrix, with elements containing atomic number, symbol and name
    for x in range(0, len(lines)):
        element_names.append(lines[x].split(","))
        
    #Gives name for atomic number
    for x in range(0,len(numbers)):
        names.append(element_names[int(numbers[x]) - 1][1]) 

    #Print coordinates to new input_coords.xyz file
    for i in range(0, N):
        for j in range(0,3):
            xyz[i][j] = list_coords[n]
            n = n + 1

        file.write(names[i] + str(round(xyz[i][0],3)) + ' ' + str(round(xyz[i][1],3)) + ' ' + str(round(xyz[i][2], 3)) + '\n')

    file.close()

    return N


def  u_PA_from_angles( atom_A, atom_B, atom_C, coords ):
    #This gives the vector in the plane A,B,C and perpendicular to A to B

    import numpy as np
    from unit_vector_N import unit_vector_N

    diff_AB = coords[atom_B,:] - coords[atom_A,:]
    norm_diff_AB = np.linalg.norm(diff_AB)
    u_AB = diff_AB / norm_diff_AB

    diff_CB = coords[atom_B,:] - coords[atom_C,:]
    norm_diff_CB = np.linalg.norm(diff_CB)
    u_CB = diff_CB / norm_diff_CB

    u_N = unit_vector_N( u_CB, u_AB )

    u_PA = np.cross(u_N,  u_AB)
    norm_PA = np.linalg.norm(u_PA)
    u_PA = u_PA /norm_PA;

    return u_PA

def force_angle_constant( atom_A, atom_B, atom_C, bond_lengths, eigenvalues, eigenvectors, coords, scaling_1, scaling_2 ):
    #Force Constant- Equation 14 of seminario calculation paper - gives force
    #constant for angle (in kcal/mol/rad^2) and equilibrium angle in degrees

    import math
    import numpy as np 
    from unit_vector_N import unit_vector_N
    
    #Vectors along bonds calculated
    diff_AB = coords[atom_B,:] - coords[atom_A,:]
    norm_diff_AB = np.linalg.norm(diff_AB)
    u_AB = diff_AB / norm_diff_AB

    diff_CB = coords[atom_B,:] - coords[atom_C,:]
    norm_diff_CB = np.linalg.norm(diff_CB)
    u_CB = diff_CB / norm_diff_CB

    #Bond lengths and eigenvalues found
    bond_length_AB = bond_lengths[atom_A,atom_B]
    eigenvalues_AB = eigenvalues[atom_A, atom_B, :]
    eigenvectors_AB = eigenvectors[0:3, 0:3, atom_A, atom_B]

    bond_length_BC = bond_lengths[atom_B,atom_C]
    eigenvalues_CB = eigenvalues[atom_C, atom_B,  :]
    eigenvectors_CB = eigenvectors[0:3, 0:3, atom_C, atom_B]

    #Normal vector to angle plane found
    u_N = unit_vector_N( u_CB, u_AB )

    u_PA = np.cross(u_N,  u_AB)
    norm_u_PA = np.linalg.norm(u_PA)
    u_PA = u_PA / norm_u_PA

    u_PC = np.cross(u_CB, u_N)
    norm_u_PC = np.linalg.norm(u_PC)
    u_PC = u_PC / norm_u_PC

    sum_first = 0 
    sum_second = 0

    #Projections of eigenvalues
    for i in range(0,3):
        eig_AB_i = eigenvectors_AB[:,i]
        eig_BC_i = eigenvectors_CB[:,i]
        sum_first = sum_first + ( eigenvalues_AB[i] * abs(dot_product(u_PA , eig_AB_i) ) ) 
        sum_second = sum_second +  ( eigenvalues_CB[i] * abs(dot_product(u_PC, eig_BC_i) ) ) 

    #Scaling due to additional angles - Modified Seminario Part
    sum_first = sum_first/scaling_1 
    sum_second = sum_second/scaling_2

    #Added as two springs in series
    k_theta = ( 1 / ( (bond_length_AB**2) * sum_first) ) + ( 1 / ( (bond_length_BC**2) * sum_second) ) 
    k_theta = 1/k_theta

    k_theta = - k_theta #Change to OPLS form

    k_theta = abs(k_theta * 0.5) #Change to OPLS form

    #Equilibrium Angle
    theta_0 = math.degrees(math.acos(np.dot(u_AB, u_CB)))

    return k_theta, theta_0

def dot_product(u_PA , eig_AB):
    x = 0     
    for i in range(0,3):
        x = x + u_PA[i] * eig_AB[i].conjugate()
    return x 

def force_constant_bond(atom_A, atom_B, eigenvalues, eigenvectors, coords):
    #Force Constant - Equation 10 of Seminario paper - gives force
    #constant for bond
    import numpy as np

    #Eigenvalues and eigenvectors calculated 
    eigenvalues_AB = eigenvalues[atom_A,atom_B,:]

    eigenvectors_AB = eigenvectors[:,:,atom_A,atom_B]

    #Vector along bond 
    diff_AB = np.array(coords[atom_B,:]) - np.array(coords[atom_A,:])
    norm_diff_AB = np.linalg.norm(diff_AB)

    unit_vectors_AB = diff_AB / norm_diff_AB
    
    k_AB = 0 

    #Projections of eigenvalues 
    for i in range(0,3):
        dot_product = abs(np.dot(unit_vectors_AB, eigenvectors_AB[:,i]))
        k_AB = k_AB + ( eigenvalues_AB[i] * dot_product )

    k_AB = -k_AB * 0.5 # Convert to OPLS form

    return k_AB




def input_data_processing_g09():
    #This function takes input data that is need from the files supplied
    #Function extracts input coords and hessian from .fchk file, bond and angle
    #lists from .log file and atom names if a z-matrix is supplied

    import numpy as np
    import os.path


    #Gets Hessian in unprocessed form and writes .xyz file too 
    [unprocessed_Hessian, N, names, coords] =  coords_from_fchk( 'lig.fchk' );

    #Gets bond and angle lists
    [bond_list, angle_list] = bond_angle_list_gaussian()

    with open("Number_to_Atom_type") as f:
        OPLS_number_to_name = f.readlines()

    OPLS_number_to_name = [x.split() for x in OPLS_number_to_name]; 

    length_hessian = 3 * N
    hessian = np.zeros((length_hessian, length_hessian))
    m = 0

    #Write the hessian in a 2D array format 
    for i in range (0,(length_hessian)):
        for j in range (0,(i + 1)):
            hessian[i][j] = unprocessed_Hessian[m]
            hessian[j][i] = unprocessed_Hessian[m]
            m = m + 1
  
    hessian = (hessian * (627.509391) )/ (0.529**2) ; #Change from Hartree/bohr to kcal/mol /ang

    # if zmat exists part here 
    atom_names = []

    for i in range(0,len(names)):
        atom_names.append(names[i].strip() + str(i + 1))

    if os.path.exists(inputfilefolder + 'Zmat.z'):    
        atom_names = []
        
        fid = open(inputfilefolder + 'Zmat.z') #Boss type Zmat
   
        tline = fid.readline()
        tline = fid.readline()

        #Find number of dummy atoms
        number_dummy = 0
        tmp = tline.split()
        
        while tmp[2] == '-1':
            number_dummy = number_dummy + 1
            tline = fid.readline()
            tmp = tline.split()

        if int(tmp[3]) < 800:
            for i in range(0,N):
                for j in range(0, len(OPLS_number_to_name)):
                    if OPLS_number_to_name[j][0] == tmp[3]:
                        atom_names.append(OPLS_number_to_name[j][1])
                
                tline = fid.readline()
                tmp = tline.split()
        else:
            #For CM1A format
            while len(tmp) < 2 or tmp[1] != 'Non-Bonded':
                tline = fid.readline()
                tmp = tline.split()
            
            tline = fid.readline()
            tline = fid.readline()
            tmp = tline.split()

            for i in range(0,N):
                atom_names.append(tmp[2])
                tline = fid.readline()
                tmp = tline.split()

        for i in range(0,N):
            if len(atom_names[i]) == 1:
                atom_names[i] = atom_names[i] + ' '
            
    return(bond_list, angle_list, coords, N, hessian, atom_names)

def coords_from_fchk( fchk_file):
    #Function extracts xyz file from the .fchk output file from Gaussian, this
    #provides the coordinates of the molecules
    import os.path
    import numpy as np

    if os.path.exists(fchk_file):
        fid = open( fchk_file, "r")
    else:
        fid_log = open( 'MSM_log', "a")
        fid_log.write('ERROR = No .fchk file found.')
        fid_log.close()
        return 0,0
        
    tline = fid.readline()

    loop = 'y'
    numbers = [] #Atomic numbers for use in xyz file
    list_coords = [] #List of xyz coordinates
    hessian = []

    #Get atomic number and coordinates from fchk 
    while tline:
        #Atomic Numbers found
        if len(tline) > 16 and (tline[0:15].strip() == 'Atomic numbers'):
            tline = fid.readline()
            while len(tline) < 17 or (tline[0:16].strip() != 'Nuclear charges'):
                tmp = (tline.strip()).split()
                numbers.extend(tmp)
                tline = fid.readline()
            
        #Get coordinates
        if len(tline) > 31 and tline[0:31].strip() == 'Current cartesian coordinates':
            tline = fid.readline()
            while len(tline) < 15 or ( tline[0:14].strip() != 'Force Field' and tline[0:17].strip() != 'Int Atom Types'and tline[0:13].strip() != 'Atom Types'):
                tmp = (tline.strip()).split()
                list_coords.extend(tmp)
                tline = fid.readline()
            N = int( float(len(list_coords)) / 3.0 ) #Number of atoms

        #Gets Hessian 
        if len(tline) > 25 and (tline[0:26].strip() == 'Cartesian Force Constants'):
            tline = fid.readline()
            
            while len(tline) < 13 or (tline[0:14].strip() != 'Dipole Moment'):
                tmp = (tline.strip()).split()
                hessian.extend(tmp)
                tline = fid.readline()

            loop = 'n'

        tline = fid.readline()

    fid.close()

    list_coords = [float(x)*float(0.529) for x in list_coords]

    #Opens the new xyz file 
    file = open( 'input_coords.xyz', "w+")
    file.write(str(N) + '\n \n')

    xyz = np.zeros((N,3))
    n = 0

    names = []

    fid_csv = open('elementlist.csv', "r")

    with fid_csv as f:
        lines = fid_csv.read().splitlines()

    element_names = []

    #Turn list in a matrix, with elements containing atomic number, symbol and name
    for x in range(0, len(lines)):
        element_names.append(lines[x].split(","))
        
    #Gives name for atomic number
    for x in range(0,len(numbers)):
        names.append(element_names[int(numbers[x]) - 1][1]) 

    #Print coordinates to new input_coords.xyz file
    for i in range(0, N):
        for j in range(0,3):
            xyz[i][j] = list_coords[n]
            n = n + 1

        file.write(names[i] + str(round(xyz[i][0],3)) + ' ' + str(round(xyz[i][1],3)) + ' ' + str(round(xyz[i][2], 3)) + '\n')

    file.close()
    return (hessian, N, names, xyz) 

def bond_angle_list_gaussian():
#This function extracts a list of bond and angles from the Gaussian .log
#file
    import os.path

    fname =  '/zmat.log'

    if os.path.isfile(fname) :
        fid = open(fname, "r")
    elif os.path.isfile( '/lig.log'):
        fid = open('/lig.log', "r")
    else:
        fid_log = open('MSM_log', "a")
        fid_log.write('ERROR - No .log file found. \n')
        fid_log.close
        return 

    tline = fid.readline()
    bond_list = []
    angle_list = []

    n = 1
    n_bond = 1
    n_angle = 1
    tmp = 'R' #States if bond or angle
    B = []

    #Finds the bond and angles from the .log file
    while tline:
        tline = fid.readline()
        #Line starts at point when bond and angle list occurs
        if len(tline) > 80 and tline[0:81].strip() == '! Name  Definition              Value          Derivative Info.                !':
            tline = fid.readline()
            tline = fid.readline()
            #Stops when all bond and angles recorded 
            while ( ( tmp[0] == 'R' ) or (tmp[0] == 'A') ):
                line = tline.split()
                tmp = line[1]
                
                #Bond or angles listed as string
                list_terms = line[2][2:-1]

                #Bond List 
                if ( tmp[0] == 'R' ): 
                    x = list_terms.split(',')
                    #Subtraction due to python array indexing at 0
                    x = [(int(i) - 1 ) for i in x]
                    bond_list.append(x)

                #Angle List 
                if ( tmp[0] == 'A' ): 
                    x = list_terms.split(',')
                    #Subtraction due to python array indexing at 0
                    x = [(int(i) - 1 ) for i in x]
                    angle_list.append(x)

                tline = fid.readline()

            #Leave loop
            tline = -1

    return(bond_list, angle_list)



def sb_file_new_parameters(inputfilefolder, filename):
#Takes new angle and bond terms and puts them into a .sb file with name
#filename_seminario.sb

    fid_angles = open(inputfilefolder + 'Average_Modified_Seminario_Angles', 'r')
    fid_bonds = open(inputfilefolder + 'Average_Modified_Seminario_Bonds', 'r')

    angles = [] 
    bonds = []

    bond_lines = fid_bonds.readline()

    while bond_lines:
        bonds.append(bond_lines.strip().split('  '))
        bond_lines = fid_bonds.readline()

    angle_lines = fid_angles.readline()

    while angle_lines:
        angles.append(angle_lines.strip().split('  '))
        angle_lines = fid_angles.readline()

    #Opens Files
    fidout = open(inputfilefolder + filename  + '_Seminario.sb', 'wt') #Script produces this file
    fidout.write('*****                         Bond Stretching and Angle Bending Parameters - July 17*****\n')

    #Prints out bonds at top of file
    for i in range(0,len(bonds)):
        if (len(bonds[i]) == 5 ):
            fidout.write( bonds[i][0] + ' ' +  bonds[i][1] + '      ' + bonds[i][2] + '        Modified Seminario Method AEAA \n')
        else:
            fidout.write( bonds[i][0] + ' ' +  bonds[i][1] + '      ' +   bonds[i][2] + '        Modified Seminario Method AEAA \n' )

    fidout.write('\n')
    fidout.write('********                        line above must be blank\n')

    #Prints out angles in middle of file
    for i in range(0,len(angles)):
        if len(angles[i]) == 8:
            fidout.write( angles[i][0] + '    ' + angles[i][1] + '       ' +  angles[i][2] + '    Modified Seminario Method AEAA \n')
        else:
            fidout.write( angles[i][0] + '     ' +  angles[i][1] + '       ' +  angles[i][2] + '        Modified Seminario Method AEAA \n')

    fidout.write('\n')
    fidout.write('\n')

    fidout.close()
    fid_angles.close()
    fid_bonds.close()


def unit_vector_N(u_BC, u_AB):
    # Calculates unit normal vector which is perpendicular to plane ABC
    import numpy as np

    cross_product = np.cross(u_BC, u_AB)
    norm_u_N = np.linalg.norm(cross_product)
    u_N = cross_product / norm_u_N
    return u_N


