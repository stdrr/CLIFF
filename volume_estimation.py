import argparse
import sys
import os
import pymesh

import bpy

ext_libs_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(ext_libs_path)

import external_modules.object_print3d_utils as op3d
from common_.import_export import import_obj, import_, register, unregister
from common_.mesh import join_mesh, make_manifold, delete_mesh
from common_.utils import verbosity_filter_func


systems_aliases = {'NONE': 'NONE', 'METRIC': 'METRIC', 'IMPERIAL': 'IMPERIAL',
                   'MM': 'METRIC', 'CM': 'METRIC', 'M': 'METRIC', 'KM': 'METRIC',
                   'IN': 'IMPERIAL', 'FT': 'IMPERIAL', 'YD': 'IMPERIAL', 'MI': 'IMPERIAL',
                   'MILLIMITERS': 'METRIC', 'CENTIMETERS': 'METRIC', 'METERS': 'METRIC', 'KILOMETERS': 'METRIC',
                   'INCHES': 'IMPERIAL', 'FEET': 'IMPERIAL', 'YARDS': 'IMPERIAL', 'MILES': 'IMPERIAL'}

units_aliases = {'MILLIMETERS': 'MILLIMETERS', 'CENTIMETERS': 'CENTIMETERS', 'METERS': 'METERS', 'KILOMETERS': 'KILOMETERS',
                 'INCHES': 'INCHES', 'FEET': 'FEET', 'YARDS': 'YARDS', 'MILES': 'MILES',
                 'MM': 'MILLIMETERS', 'CM': 'CENTIMETERS', 'M': 'METERS', 'M': 'KILOMETERS',
                 'IN': 'INCHES', 'FT': 'FEET', 'YD': 'YARDS', 'MI': 'MILES'}



def compute_volume_of_selected_mesh_blender(system:str, unit:str):
    """
    Computes the volume of the selected mesh.
    The computation entirely relies on the Blender's built-in operator bpy.ops.mesh.print3d_info_volume().
    Some additional checks on the measurement system and unit of measurement are performed.
    """

    # Check the input
    system = systems_aliases.get(system.upper(), 'METRIC')
    unit = units_aliases.get(unit.upper(), 'METERS')
    # Set the unit of measurement
    bpy.context.scene.unit_settings.system = system
    bpy.context.scene.unit_settings.length_unit = unit
    # Compute the volume of the selected mesh
    bpy.ops.mesh.print3d_info_volume()
    
    
def compute_volume_from_obj_blender(src_file:str):
    """
    Imports the mesh and computes the volume of the selected mesh.
    The computation entirely relies on the Blender's built-in operator bpy.ops.mesh.print3d_info_volume().
    Some additional checks on the measurement system and unit of measurement are performed.
    """
    system = 'METRIC'
    unit = 'METERS'
    # Register the classes and operators
    op3d.register()
    # Import the mesh
    import_obj(src_file)
    bpy.ops.object.select_all(action='SELECT')
    compute_volume_of_selected_mesh_blender(system, unit)
    volume = op3d.report.info()[0][0]
    volume = float(volume.split(' ')[1])
    op3d.unregister()
    return volume
    
    
def compute_volume_of_selected_mesh_pymesh(mesh:pymesh.Mesh):
    """
    Computes the volume of the selected mesh.
    The computation entirely relies on the pymesh module.
    """
    return mesh.volume
    
    
def compute_dimensions_of_selected_mesh_blender():
    return bpy.context.active_object.dimensions


def compute_dimensions_of_selected_mesh_pymesh(mesh:pymesh.Mesh):
    return mesh.bbox[1] - mesh.bbox[0]


def main_blender(src_file:str, args):
    """
    Main function.
    """
    print('Using blender')

    # Register the classes and operators
    op3d.register()

    mesh_name = args.mesh_name
    system = args.system
    unit = args.unit

    # Import the mesh
    is_ydd = src_file.endswith('.ydd.xml')
    if is_ydd:
        register(src_file)
        import_fun = import_
    else: 
        import_fun = import_obj
    verbosity_filter_func(args.verbosity, import_fun, src_file)
    # Delete the unnecessary meshes (hair, accessories, objects, jewels, etc.)
    verbosity_filter_func(args.verbosity, delete_mesh, ['hair', 'accs', 'decl', 'task'])
    # Make the mesh manifold
    verbosity_filter_func(args.verbosity, make_manifold)
    # Select the mesh
    verbosity_filter_func(args.verbosity, join_mesh, mesh_name)
    # Compute the volume of the selected mesh
    verbosity_filter_func(args.verbosity, compute_volume_of_selected_mesh_blender, system, unit)
    # Get the volume value
    volume = op3d.report.info()[0][0]
    # Compute the dimensions of the selected mesh
    dimensions = verbosity_filter_func(args.verbosity, compute_dimensions_of_selected_mesh_blender)
    # Print the volume value and the dimensions
    output = '{file_name} [blender] - {volume} (X:{x:.4f}, Y:{y:.4f}, Z:{z:.4f})cm\n'.format(
        file_name=os.path.basename(src_file), volume=volume, 
        x=dimensions[0]*100, y=dimensions[1]*100, z=dimensions[2]*100)
    if args.log_file is not None:
        with open(args.log_file, 'a') as f:
            f.write(output)
    else:        
        print(output)

    op3d.unregister()
    if is_ydd:
        unregister(is_ydd)
        

def main_pymesh(src_file:str, args):
    print('Using pymesh')
    # Load the mesh
    mesh = pymesh.load_mesh(src_file)
    # Compute the volume
    volume = compute_volume_of_selected_mesh_pymesh(mesh)
    # Compute the dimensions
    dimensions = compute_dimensions_of_selected_mesh_pymesh(mesh)
    output = '{file_name} [pymesh] - Volume:{volume:.4f}m³ (X:{x:.4f}, Y:{y:.4f}, Z:{z:.4f})cm\n'.format(
        file_name=os.path.basename(src_file), volume=volume, 
        x=dimensions[2]*100, y=dimensions[1]*100, z=dimensions[0]*100)
    if args.log_file is not None:
        with open(args.log_file, 'a') as f:
            f.write(output)
    else:        
        print(output)



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Volume estimator.')
    parser.add_argument('--library', type=str, default='blender', help='The library to be used for the computation. (blender, pymesh)')
    parser.add_argument('--mesh_name', '-m', type=str, default='ALL', help='The name of the mesh to be selected.')
    parser.add_argument('--system', '-s', type=str, default='METRIC', help='The measurement system to be used.')
    parser.add_argument('--unit', '-u', type=str, default='METERS', help='The unit of measurement to be used.')
    parser.add_argument('--verbosity', '-v', type=int, default=0, help='The verbosity level. 0: no output, 1: only errors, 2: all')
    parser.add_argument('--log_file', type=str, default=None, help='Path to log file.')
    subparsers = parser.add_subparsers()
    single_file_parser = subparsers.add_parser('single_file', help='Use one file.')
    single_file_parser.add_argument('-s', '--source', type=str, help='Path to source YDD.')
    multiple_files_parser = subparsers.add_parser('multiple_files', help='Use multiple files.')
    multiple_files_parser.add_argument('-d', '--dir', type=str, help='Path to the directory.')
    multiple_files_parser.add_argument('-l', '--list_of_files', type=str,
                                       help='Path to a file containing the paths to multiple files.')
    args = parser.parse_args()
    
    main = main_blender if args.library == 'blender' else main_pymesh
        
    if 'list_of_files' in args:
        # Use multiple files
        with open(args.list_of_files, 'r') as f:
            files = f.read().splitlines()
            
        root_dir = args.dir
            
        files = [os.path.join(root_dir, os.path.dirname(args.list_of_files).split('/')[-1], f) for f in files]
        
        for file in files:
            main(file, args)
    else:
        main(args.source, args)