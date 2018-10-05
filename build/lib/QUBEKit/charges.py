#!/usr/bin/env python

def charge_gen(input_cube, ddec_version, chargemol_path):

    if ddec_version == 6:

        file = open('job_control', 'w+')
        file.write('<periodicity along A, B and C vectors>\n.true.\n.true.\n.true.\n</periodicity along A, B and C vectors>')

        file.write('\n\n<atomic densities directory complete path>\n{}'.format(chargemol_path))
        file.write('\n</atomic densities directory complete path>')

        file.write('\n\n<input filename>\n{}\n</input filename>'.format(input_cube))

        file.close()

    elif ddec_version == 3:
        # TODO handle DDEC3 (and maybe other versions).
        pass

    else:

        print('invalid DDEC version, please use versions 3 or 6.')
