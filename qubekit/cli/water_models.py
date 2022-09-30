from openmm import unit

water_models = {
    "tip3p": {
        "Bonds": [
            {
                "smirks": "[#1:1]-[#8X2H2+0:2]-[#1]",
                "k": 1087.053566377 * unit.kilocalorie_per_mole / unit.angstroms**2,
                "length": 0.9572 * unit.angstroms,
            }
        ],
        "Angles": [
            {
                "smirks": "[#1:1]-[#8X2H2+0:2]-[#1:3]",
                "k": 130.181232192 * unit.kilocalorie_per_mole / unit.radian**2,
                "angle": 110.3538806181 * unit.degree,
            }
        ],
        "Constraints": [
            {
                "smirks": "[#1:1]-[#8X2H2+0:2]-[#1]",
                "id": "c-tip3p-H-O",
                "distance": 0.9572 * unit.angstroms,
            },
            {
                "smirks": "[#1:1]-[#8X2H2+0]-[#1:2]",
                "id": "c-tip3p-H-O-H",
                "distance": 1.5139006545247014 * unit.angstroms,
            },
        ],
        "LibraryCharges": [
            {
                "smirks": "[#1]-[#8X2H2+0:1]-[#1]",
                "charge1": -0.834 * unit.elementary_charge,
                "id": "q-tip3p-O",
            },
            {
                "smirks": "[#1:1]-[#8X2H2+0]-[#1]",
                "charge1": 0.417 * unit.elementary_charge,
                "id": "q-tip3p-H",
            },
        ],
        "Nonbonded": [
            {
                "smirks": "[#1]-[#8X2H2+0:1]-[#1]",
                "epsilon": 0.1521 * unit.kilocalorie_per_mole,
                "id": "n-tip3p-O",
                "sigma": 3.1507 * unit.angstroms,
            },
            {
                "smirks": "[#1:1]-[#8X2H2+0]-[#1]",
                "epsilon": 0 * unit.kilocalorie_per_mole,
                "id": "n-tip3p-H",
                "sigma": 1 * unit.angstroms,
            },
        ],
    },
    "tip4p-fb": {
        "Bonds": [
            {
                "smirks": "[#1:1]-[#8X2H2+0:2]-[#1]",
                "k": 1087.053566377 * unit.kilocalorie_per_mole / unit.angstroms**2,
                "length": 0.9572 * unit.angstroms,
            }
        ],
        "Angles": [
            {
                "smirks": "[#1:1]-[#8X2H2+0:2]-[#1:3]",
                "k": 130.181232192 * unit.kilocalorie_per_mole / unit.radian**2,
                "angle": 110.3538806181 * unit.degree,
            }
        ],
        "Constraints": [
            {
                "smirks": "[#1:1]-[#8X2H2+0:2]-[#1]",
                "id": "c-tip4p-fb-H-O",
                "distance": 0.9572 * unit.angstroms,
            },
            {
                "smirks": "[#1:1]-[#8X2H2+0]-[#1:2]",
                "id": "c-tip4p-fb-H-O-H",
                "distance": 1.5139006545247014 * unit.angstroms,
            },
        ],
        "LibraryCharges": [
            {
                "smirks": "[#1]-[#8X2H2+0:1]-[#1]",
                "charge1": 0 * unit.elementary_charge,
                "id": "q-tip4p-fb-O",
            },
            # charge is set to zero as the v-site will move the charge
            {
                "smirks": "[#1:1]-[#8X2H2+0]-[#1]",
                "charge1": 0.52587 * unit.elementary_charge,
                "id": "q-tip4p-fb-H",
            },
        ],
        "Nonbonded": [
            {
                "smirks": "[#1]-[#8X2H2+0:1]-[#1]",
                "epsilon": 0.179082 * unit.kilocalorie_per_mole,
                "id": "n-tip4p-fb-O",
                "sigma": 3.1655 * unit.angstroms,
            },
            {
                "smirks": "[#1:1]-[#8X2H2+0]-[#1]",
                "epsilon": 0 * unit.kilocalorie_per_mole,
                "id": "n-tip4p-fb-H",
                "sigma": 1 * unit.angstroms,
            },
        ],
        "LocalCoordinateVirtualSites": [
            {
                "smirks": "[#1:2]-[#8X2H2+0:1]-[#1:3]",
                "type": "local",
                "name": "q-tip4p-fb-v",
                "x_local": 0.010527 * unit.nanometers,
                "y_local": 0.0 * unit.nanometers,
                "z_local": 0.0 * unit.nanometers,
                "match": "once",
                "charge": -1.05174 * unit.elementary_charge,
                "sigma": 1 * unit.nanometers,
                "epsilon": 0.0 * unit.kilojoule_per_mole,
            }
        ],
    },
}
