#!/bin/python3

# ===================================================================================
# Name.......: SASP - Shortest Accessible Surface Distance Calculator
# Authors....: Joaquin Algorta
# Contact....: joaquin.algorta@mpimp-golm.mpg.de
# Description: Pipeline to compute the shortest accessible path between two points
#              around a molecular surface, avoiding the mesh interior. Useful for
#              analyzing protein-protein interfaces, ligand accessibility, and
#              other structural biology applications.
# ===================================================================================

import argparse
from functions import *  # Import all functions from functions.py


def main(input_pdb, start_point, end_point, max_distance=1.9, resolution=1, output_dir=None, just_dist=False):
    # Step 1: Generate MSMS files
    mesh = run_msms(input_pdb, output_dir)
    print("Number of vertices:", mesh["num_vertexes"])
    print("Number of triangles:", mesh["num_triangles"])
    print("Number of spheres:", mesh["num_spheres"])

    # Create the mesh using trimesh
    vertices = np.array(mesh['vertices'])
    faces = np.array(mesh['faces'])
    your_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    # Step 2: Define grid parameters, considering start and end points
    # Get bounds that include ALL mesh vertices, start point, and end point
    all_points = np.vstack([vertices, start_point.reshape(1, -1), end_point.reshape(1, -1)])
    min_bounds = all_points.min(axis=0) - 2  # Add 2 Å padding beyond everything
    max_bounds = all_points.max(axis=0) + 2  # Add 2 Å padding beyond everything

    points_outside_mesh = generate_3d_grid_outside_mesh2(your_mesh, min_bounds, max_bounds, resolution)
    
    print("Points outside mesh:", points_outside_mesh.shape)

    # Combine mesh vertices and points outside
    combined_points = np.vstack([vertices, points_outside_mesh])

    # Step 3: Build adjacency list
    adjacency_list = build_adjacency_list(combined_points, max_distance)

    print(start_point,end_point)

    # Find indices of start and end points
    start_idx = np.argmin(np.linalg.norm(combined_points - start_point, axis=1))
    end_idx = np.argmin(np.linalg.norm(combined_points - end_point, axis=1))

    euclidean_distance = np.linalg.norm(combined_points[start_idx] - combined_points[end_idx])

    # Step 4: Find the shortest path using Dijkstra's algorithm
    path_indices = astar_shortest_path(start_idx, end_idx, adjacency_list, combined_points)
    shortest_path_points = combined_points[path_indices]

    # Step 5, refine the path in the forward and reverse direction
    refined_forward_path = refine_path_forward(shortest_path_points, your_mesh)
    refined_reverse_path = refine_path_reverse(refined_forward_path, your_mesh)

    #refined_reverse_path = best_refine_path(refined_reverse_path, your_mesh) # Refines all the possible points - very slow

    _min_dist = calculate_path_distance(refined_reverse_path)

    print(f"Final distance: {_min_dist}")

    # save the final distance to a text file
    with open(os.path.join(output_dir, 'SASP.txt'), 'w') as f:
        f.write(str(_min_dist) + '\n')
    with open(os.path.join(output_dir, 'euclid.txt'), 'w') as f:
        f.write(str(euclidean_distance)+'\n')

    # Step 6: Visualize the results
    if not just_dist:
        visualize_mesh_outside_points_results(points_outside_mesh, your_mesh, refined_reverse_path)

    # Step 7: Write the PML file
    write_pml_file(refined_reverse_path, input_pdb, output_dir=output_dir)
        
    return _min_dist, euclidean_distance

def main_no_contacts(input_pdb, start_point, end_point, max_distance=1.9, resolution=1, output_dir=None, just_dist=False):
    # Step 1: Generate MSMS files
    mesh = run_msms_separate_chains(input_pdb, output_dir)
    print("Number of vertices:", mesh["num_vertexes"])
    print("Number of triangles:", mesh["num_triangles"])

    # Create the mesh using trimesh
    vertices = np.array(mesh['vertices'])
    faces = np.array(mesh['faces'])
    your_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    # Step 2: Define grid parameters
    min_bounds = vertices.min(axis=0) - 1
    max_bounds = vertices.max(axis=0) + 1

    points_outside_mesh = generate_3d_grid_outside_mesh2(your_mesh, min_bounds, max_bounds, resolution)
    
    print("Points outside mesh:", points_outside_mesh.shape)

    # Combine mesh vertices and points outside
    combined_points = np.vstack([vertices, points_outside_mesh])

    # Step 3: Build adjacency list
    adjacency_list = build_adjacency_list(combined_points, max_distance)

    print(start_point,end_point)

    # Find indices of start and end points
    start_idx = np.argmin(np.linalg.norm(combined_points - start_point, axis=1))
    end_idx = np.argmin(np.linalg.norm(combined_points - end_point, axis=1))

    euclidean_distance = np.linalg.norm(combined_points[start_idx] - combined_points[end_idx])

    # Step 4: Find the shortest path using Dijkstra's algorithm
    path_indices = astar_shortest_path(start_idx, end_idx, adjacency_list, combined_points)
    shortest_path_points = combined_points[path_indices]

    # Step 5, refine the path in the forward and reverse direction
    refined_forward_path = refine_path_forward(shortest_path_points, your_mesh)
    refined_reverse_path = refine_path_reverse(refined_forward_path, your_mesh)

    #refined_reverse_path = best_refine_path(refined_reverse_path, your_mesh)  # Refines all the possible points - very slow

    _min_dist = calculate_path_distance(refined_reverse_path)

    print(f"Final distance: {_min_dist}")

    # save the final distance to a text file
    with open(os.path.join(output_dir, 'SASP.txt'), 'w') as f:
        f.write(str(_min_dist) + '\n')
    with open(os.path.join(output_dir, 'euclid.txt'), 'w') as f:
        f.write(str(euclidean_distance)+'\n')

    # Step 6: Visualize the results
    if not just_dist:
        visualize_mesh_outside_points_results(points_outside_mesh, your_mesh, refined_reverse_path)

    # Step 7: Write the PML file
    write_pml_file(refined_reverse_path, input_pdb, output_dir=output_dir)

    return _min_dist, euclidean_distance



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute shortest accessible path avoiding mesh interior.")
    
    parser.add_argument("input_pdb", type=str, help="Path to the input PDB file.")
    parser.add_argument("start_point", type=float, nargs=3, help="Start point coordinates: x y z")
    parser.add_argument("end_point", type=float, nargs=3, help="End point coordinates: x y z")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save outputs (default: 'output').")
    parser.add_argument("--max_distance", type=float, default=1.9, help="Maximum distance to connect nodes (default: 1.9).")
    parser.add_argument("--resolution", type=float, default=1.0, help="Grid resolution (default: 1.0).")
    parser.add_argument("--just_dist", action="store_true", help="Only compute and save distances without visualization.")

    args = parser.parse_args()

    # Auto-detect based on chain contacts
    use_main, num_chains = detect_chain_contacts(args.input_pdb)
    
    # Run the appropriate function
    if use_main:
        min_dist, euclid = main(
            input_pdb=args.input_pdb,
            start_point=np.array(args.start_point),
            end_point=np.array(args.end_point),
            max_distance=args.max_distance,
            resolution=args.resolution,
            output_dir=args.output_dir,
            just_dist=args.just_dist
        )
    else:
        min_dist, euclid = main_no_contacts(
            input_pdb=args.input_pdb,
            start_point=np.array(args.start_point),
            end_point=np.array(args.end_point),
            max_distance=args.max_distance,
            resolution=args.resolution,
            output_dir=args.output_dir,
            just_dist=args.just_dist
        )

#Example usage:
# python run_shortest_path.py inputs/6r2g.pdb  30.726  43.518  90  30.726  43.518  10 --output_dir outputs

