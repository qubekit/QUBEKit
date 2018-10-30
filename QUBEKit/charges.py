#!/usr/bin/env python


# Added to engines
def charge_gen(ddec_version, chargemol_path):
    """Given a DDEC version (from the defaults), this function writes the job file for chargemol."""

    if ddec_version != 6 or ddec_version != 3:
        print('Invalid DDEC version given, running with default version 6.')
        ddec_version = 6

    with open('job_control.txt', 'w+') as file:

        file.write('<net charge>\n0.0\n</net charge>')

        file.write('\n\n<periodicity along A, B and C vectors>\n.false.\n.false.\n.false.')
        file.write('\n</periodicity along A, B and C vectors>')

        file.write('\n\n<atomic densities directory complete path>\n{}/atomic_densities/'.format(chargemol_path))
        file.write('\n</atomic densities directory complete path>')

        file.write('\n\n<charge type>\nDDEC{}\n</charge type>'.format(ddec_version))

        file.write('\n\n<compute BOs>\n.true.\n</compute BOs>')
