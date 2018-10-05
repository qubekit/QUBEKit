#!/usr/bin/env python


def charge_gen(input_cube, ddec_version, chargemol_path):

    if ddec_version == 6:

        file = open('job_control.txt', 'w+')
        file.write('<periodicity along A, B and C vectors>\n.false.\n.false.\n.false.\n</periodicity along A, B and C vectors>')

        file.write('\n\n<atomic densities directory complete path>\n{}/atomic_densities/'.format(chargemol_path))
        file.write('\n</atomic densities directory complete path>')

        file.write('\n\n<input filename>\n{}\n</input filename>'.format(input_cube))

        file.write('\n\n<charge type>\nDDEC{}\n</charge type>'.format(ddec_version))

        file.write('\n\n<compute BOs>\n.True.\n</compute BOs>')

        file.write('\n\n<density format>\ne_per_angs3\n</density format>')

        file.close()

    elif ddec_version == 3:

        # TODO handle DDEC3 (and maybe other versions).
        pass

    else:

        print('invalid DDEC version, please use version 3 or 6.')
