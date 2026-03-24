import subprocess
import plotly.graph_objects as go
from scipy.spatial import KDTree
import heapq
import os
import numpy as np
import trimesh


def run_msms(input_pdb, output_dir=None):
    """
    Runs MSMS to generate surface mesh.
    """
    if os.path.dirname(input_pdb):
        pdb_code = os.path.basename(input_pdb).rsplit('.', 1)[0]
    else:
        pdb_code = input_pdb.replace(".pdb", "")

    if output_dir is None:
        output_dir = pdb_code
    os.makedirs(output_dir, exist_ok=True)

    xyzr_path = f"{output_dir}/{pdb_code}.xyzr"
    vert_path = f"{output_dir}/{pdb_code}.vert"
    face_path = f"{output_dir}/{pdb_code}.face"

    # Execute pdb_to_xyzr and msms
    xyzr_cmd = f"pdb_to_xyzr {input_pdb} > {xyzr_path}"
    subprocess.run(xyzr_cmd, shell=True, check=True)
    msms_cmd = f"msms.x86_64Linux2.2.6.1 -if {xyzr_path} -of {output_dir}/{pdb_code} -probe_radius 1.4 -density 1"
    subprocess.run(msms_cmd, shell=True, check=True)

    """
    Parse the .vert file to extract vertex coordinates.
    """
    vertices = []
    num_vertexes, num_triangles, num_spheres, triangulation_density, probe_sphere_radius = None, None, None, None, None

    with open(f'{output_dir}/{pdb_code}.vert', 'r') as f:
        for line in f:
            if not line.strip() or line.strip().startswith('#'):
                continue
            parts = line.split()
            if len(parts) == 4:
                num_vertexes, num_spheres, triangulation_density, probe_sphere_radius = int(parts[0]), int(
                    parts[1]), float(parts[2]), float(parts[3])
            if len(parts) > 4:
                vertices.append([float(parts[0]), float(parts[1]), float(parts[2])])

    """
    Parse the .face file to extract triangle vertex indices that form triangles.
    """
    faces = []
    with open(f'{output_dir}/{pdb_code}.face', 'r') as f:
        for line in f:
            if not line.strip() or line.strip().startswith('#'):
                continue
            parts = line.split()
            if len(parts) == 4:
                num_triangles, num_spheres, triangulation_density, probe_sphere_radius = int(parts[0]), int(
                    parts[1]), float(parts[2]), float(parts[3])
            if len(parts) > 4:
                faces.append([int(parts[0]) - 1, int(parts[1]) - 1, int(parts[2]) - 1])  # Convert to 0-based

    # Remove MSMS intermediate files
    for path in [xyzr_path, vert_path, face_path]:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass

    """
    Create a mesh representation using vertices and faces.
    """
    return {"vertices": vertices, "faces": faces, "num_vertexes": num_vertexes, "num_triangles": num_triangles,
            "num_spheres": num_spheres, "triangulation_density": triangulation_density,
            "probe_sphere_radius": probe_sphere_radius}


def run_msms_separate_chains(input_pdb, output_dir=None):
    """
    Separates chains in a PDB file, runs MSMS for each chain, and combines the results into a single surface mesh.
    """

    # Define output directory
    pdb_code = os.path.splitext(os.path.basename(input_pdb))[0]
    if output_dir is None:
        output_dir = pdb_code
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Separate chains and save to individual PDB files
    chains = {}
    with open(input_pdb, 'r') as pdb_file:
        for line in pdb_file:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                chain_id = line[21]  # Chain ID is at column 22 (0-based index 21)
                if chain_id not in chains:
                    chains[chain_id] = []
                chains[chain_id].append(line)

    chain_files = []
    for chain_id, lines in chains.items():
        chain_file = os.path.join(output_dir, f"{pdb_code}_chain_{chain_id}.pdb")
        with open(chain_file, 'w') as chain_output:
            chain_output.writelines(lines)
        chain_files.append(chain_file)

    # Step 2: Convert each chain to XYZR format and run MSMS
    vert_files = []
    face_files = []
    for chain_file in chain_files:
        chain_id = os.path.basename(chain_file).split('_')[-1].split('.')[0]
        chain_xyzr = os.path.join(output_dir, f"{pdb_code}_chain_{chain_id}.xyzr")
        chain_vert = os.path.join(output_dir, f"{pdb_code}_chain_{chain_id}.vert")
        chain_face = os.path.join(output_dir, f"{pdb_code}_chain_{chain_id}.face")

        try:
            # Convert PDB to XYZR
            xyzr_cmd = f"pdb_to_xyzr {chain_file} > {chain_xyzr}"
            subprocess.run(xyzr_cmd, shell=True, check=True)

            # Run MSMS
            msms_cmd = f"msms -if {chain_xyzr} -of {os.path.join(output_dir, f'{pdb_code}_chain_{chain_id}')} -probe_radius 1.4 -density 1"
            subprocess.run(msms_cmd, shell=True, check=True)

            vert_files.append(chain_vert)
            face_files.append(chain_face)
        except subprocess.CalledProcessError as e:
            print(f"Error running MSMS for chain {chain_id}: {e}")

    # Step 3: Combine vertices and faces
    combined_vertices = []
    combined_faces = []
    vertex_offset = 0

    for vert_file, face_file in zip(vert_files, face_files):
        # Parse .vert file to extract vertex coordinates
        with open(vert_file, 'r') as vf:
            for line in vf:
                if not line.strip() or line.strip().startswith('#'):
                    continue
                parts = line.split()
                if len(parts) == 4:  # Header information
                    num_vertexes, num_spheres, triangulation_density, probe_sphere_radius = int(parts[0]), int(
                        parts[1]), float(parts[2]), float(parts[3])
                elif len(parts) > 4:  # Vertex coordinates
                    combined_vertices.append([float(parts[0]), float(parts[1]), float(parts[2])])

        # Parse .face file to extract triangle indices
        with open(face_file, 'r') as ff:
            for line in ff:
                if not line.strip() or line.strip().startswith('#'):
                    continue
                parts = line.split()
                if len(parts) == 4:  # Header information
                    num_triangles, num_spheres, triangulation_density, probe_sphere_radius = int(parts[0]), int(
                        parts[1]), float(parts[2]), float(parts[3])
                elif len(parts) > 4:  # Triangle indices
                    combined_faces.append([
                        int(parts[0]) - 1 + vertex_offset,
                        int(parts[1]) - 1 + vertex_offset,
                        int(parts[2]) - 1 + vertex_offset
                    ])

        # Update vertex offset for the next chain
        vertex_offset = len(combined_vertices)

    # Cleanup intermediate files
    for chain_file in chain_files:
        os.remove(chain_file)
    for vert_file in vert_files:
        os.remove(vert_file)
    for face_file in face_files:
        os.remove(face_file)

    # Return the combined surface data
    return {
        "vertices": combined_vertices,
        "faces": combined_faces,
        "num_vertexes": len(combined_vertices),
        "num_triangles": len(combined_faces),
    }


def generate_3d_grid_outside_mesh(mesh, min_bounds, max_bounds, resolution):
    # Generate grid points along one axis
    x = np.arange(min_bounds[0], max_bounds[0], resolution)
    y = np.arange(min_bounds[1], max_bounds[1], resolution)
    z = np.arange(min_bounds[2], max_bounds[2], resolution)

    # Prepare output container
    outside_points = []

    # Initialize the ray intersector (no external dependency)
    intersector = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)

    # Loop through grid slices along one axis (e.g., z-axis)
    for yi in y:
        for zi in z:
            # Generate ray origins along the x-axis for this slice
            ray_origins = np.array([[x_start, yi, zi] for x_start in x])
            ray_directions = np.array([[1, 0, 0]] * len(ray_origins))  # Rays along x-axis

            # Perform ray-mesh intersection test
            ray_hits = intersector.intersects_any(ray_origins, ray_directions)

            # Filter points that do not intersect the mesh
            for i, hit in enumerate(ray_hits):
                if not hit:  # If ray does not hit the mesh, add the point
                    outside_points.append(ray_origins[i])

    return np.array(outside_points)


def generate_3d_grid_outside_mesh_optimized(mesh, min_bounds, max_bounds, resolution):
    x = np.arange(min_bounds[0], max_bounds[0], resolution)
    y = np.arange(min_bounds[1], max_bounds[1], resolution)
    z = np.arange(min_bounds[2], max_bounds[2], resolution)

    # Generate ray origins for one slice along x-axis
    yz_plane = np.array(np.meshgrid(y, z, indexing='ij')).reshape(2, -1).T
    ray_origins = np.hstack([np.full((len(yz_plane), 1), min_bounds[0]), yz_plane])

    # Rays point in the positive x-direction
    ray_directions = np.array([1, 0, 0])

    # Initialize intersector
    intersector = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)

    # Batch process rays for the entire yz-plane
    ray_hits = intersector.intersects_any(ray_origins, np.tile(ray_directions, (len(ray_origins), 1)))

    # Select points where the ray does not hit
    valid_points = ray_origins[~ray_hits]

    # Expand results across all x-coordinates
    outside_points = []
    for xi in x:
        outside_points.append(valid_points + [xi, 0, 0])
    outside_points = np.vstack(outside_points)

    return outside_points


def generate_3d_grid_outside_mesh2(mesh, min_bounds, max_bounds, resolution):
    # Calculate the grid points
    x = np.arange(min_bounds[0], max_bounds[0], resolution)
    y = np.arange(min_bounds[1], max_bounds[1], resolution)
    z = np.arange(min_bounds[2], max_bounds[2], resolution)
    grid_points = np.array(np.meshgrid(x, y, z, indexing='ij')).reshape(3, -1).T

    # Initialize intersector
    intersector = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)

    # Initialize ray origins and directions
    ray_origins = grid_points
    ray_directions = np.zeros_like(ray_origins)
    ray_directions[:, 0] = 1  # rays point in the positive x-direction

    # Batch process rays
    ray_hits = intersector.intersects_any(ray_origins, ray_directions)

    # Select points where the ray does not hit
    outside_points = grid_points[~ray_hits]

    return outside_points


def build_adjacency_list(combined_points, max_distance):
    """
    Build adjacency list for combined points (mesh vertices + outside points).
    """
    kdtree = KDTree(combined_points)
    adjacency_list = {i: [] for i in range(len(combined_points))}

    for i, point in enumerate(combined_points):
        # Find neighbors within max_distance
        neighbors = kdtree.query_ball_point(point, r=max_distance)
        for neighbor in neighbors:
            if neighbor != i:  # Avoid self-loops
                adjacency_list[i].append(neighbor)
    return adjacency_list


def dijkstra_shortest_path(start_idx, end_idx, adjacency_list, combined_points):
    """
    Perform Dijkstra's algorithm to find the shortest path between two points in an adjacency list.
    """
    # Initialize distances and predecessors
    distances = {i: float('inf') for i in adjacency_list}
    distances[start_idx] = 0
    predecessors = {i: None for i in adjacency_list}

    # Priority queue for the nodes to visit
    queue = [(0.0, start_idx)]  # (distance, node)

    while queue:
            current_distance, current_node = heapq.heappop(queue)

            if current_node == end_idx:
                break  # Exit when the end node is reached

            for neighbor in adjacency_list[current_node]:
                # Calculate the distance to each neighbor
                new_distance = float(current_distance + np.linalg.norm(combined_points[current_node] - combined_points[neighbor]))
                if new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    predecessors[neighbor] = current_node
                    heapq.heappush(queue, (new_distance, neighbor))

    # Reconstruct the shortest path
    path = []
    current_node = end_idx
    while current_node is not None:
        path.append(current_node)
        current_node = predecessors[current_node]

    return path[::-1]  # Return reversed path


def astar_shortest_path(start_idx, end_idx, adjacency_list, combined_points):
    """
    Perform A* algorithm to find the shortest path between two points in an adjacency list.
    """
    # Initialize distances and predecessors
    distances = {i: float('inf') for i in adjacency_list}  # Distance from start to node
    distances[start_idx] = 0
    predecessors = {i: None for i in adjacency_list}

    # Priority queue with the heuristic included
    queue = [(0.0, start_idx)]  # (f_score, node), where f_score = g_score + h_score

    while queue:
        current_f_score, current_node = heapq.heappop(queue)

        # Stop if we reach the target
        if current_node == end_idx:
            break

        for neighbor in adjacency_list[current_node]:
            # Compute tentative g_score (distance from start to neighbor)
            g_score = float(distances[current_node] + np.linalg.norm(
                combined_points[current_node] - combined_points[neighbor]))

            if g_score < distances[neighbor]:
                distances[neighbor] = g_score
                predecessors[neighbor] = current_node

                # f_score = g_score + h_score (heuristic: straight-line distance to end node)
                h_score = np.linalg.norm(combined_points[neighbor] - combined_points[end_idx])
                f_score = float(g_score + h_score)
                heapq.heappush(queue, (f_score, neighbor))

    # Reconstruct the shortest path
    path = []
    current_node = end_idx
    while current_node is not None:
        path.append(current_node)
        current_node = predecessors[current_node]

    return path[::-1]  # Return reversed path


# Step 5: Visualize the results
def visualize_mesh_outside_points_results(points_outside_mesh, mesh, shortest_path_points):
    # Mesh Trace
    mesh_trace = go.Mesh3d(
        x=mesh.vertices[:, 0],
        y=mesh.vertices[:, 1],
        z=mesh.vertices[:, 2],
        i=mesh.faces[:, 0],
        j=mesh.faces[:, 1],
        k=mesh.faces[:, 2],
        color='lightblue',
        opacity=0.5,
        name='Mesh'
    )

    # Outside Points Trace
    outside_trace = go.Scatter3d(
        x=points_outside_mesh[:, 0],
        y=points_outside_mesh[:, 1],
        z=points_outside_mesh[:, 2],
        mode='markers',
        marker=dict(size=2, color='red'),
        name='Points Outside Mesh'
    )

    surface_trace = go.Scatter3d(
        x=mesh.vertices[:, 0],
        y=mesh.vertices[:, 1],
        z=mesh.vertices[:, 2],
        mode='markers',
        marker=dict(size=2, color='darkblue'),
        name='surface points'
    )

    path_trace = go.Scatter3d(
        x=shortest_path_points[:, 0],
        y=shortest_path_points[:, 1],
        z=shortest_path_points[:, 2],
        mode='lines+markers',
        marker=dict(size=6, color='green'),
        line=dict(color='green', width=6),
        name='Shortest Path'
    )

    fig = go.Figure(data=[mesh_trace, path_trace])

    # Update Layout
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
        ),
        title=f"Shortest path between the 2 points"
    )

    # Show the plot
    fig.show()


def calculate_path_distance(path_points):
    """
    Calculate the total distance of the shortest path by summing the distances between consecutive points.
    """
    total_distance = 0.0
    for i in range(1, len(path_points)):
        # Calculate Euclidean distance between consecutive points
        distance = np.linalg.norm(path_points[i] - path_points[i - 1])
        total_distance += distance
    return total_distance


def refine_path_forward(path_points, mesh):
    """
    Refine the path by eliminating unnecessary intermediate points if the direct line
    from a point to the end does not intersect the mesh. Skip rays entirely inside the mesh.
    """
    refined_path = [path_points[0]]  # Start with the first point
    intersector = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)

    i = 0
    while i < len(path_points) - 1:
        start_point = path_points[i]  # Current start point
        end_point = path_points[-1]  # Always check to the final destination

        # Subdivide the ray segment to check points along the segment
        num_steps = 5  # Subdivision steps
        segment_points = np.linspace(start_point, end_point, num_steps)

        # Check if all points along the segment lie inside the mesh
        inside_status = mesh.contains(segment_points)

        if np.any(inside_status):  # Entire segment is inside the mesh
            next_point = path_points[i + 1]  # Move to the next intermediate point
            refined_path.append(next_point)
        else:
            # Perform ray intersection check
            ray_origin = np.array([start_point])  # Shape (1, 3)
            ray_direction = np.array([end_point - start_point])  # Shape (1, 3)
            segment_length = np.linalg.norm(end_point - start_point)  # Length of the segment (scalar)

            # Find intersection locations (exact points)
            locations, index_ray, _ = intersector.intersects_location(
                ray_origins=ray_origin,
                ray_directions=ray_direction
            )

            # Check if any intersection occurs within the segment length
            if locations.shape[0] > 0:  # There are intersections
                distances = np.linalg.norm(locations - start_point, axis=1)  # Calculate distances to intersections

                if np.any(distances < segment_length):  # Intersection before reaching the end point
                    next_point = path_points[i + 1]  # Move to the next intermediate point
                    refined_path.append(next_point)
                else:
                    # No intersection, add the endpoint
                    refined_path.append(end_point)
                    break
            else:
                # No intersection, add the endpoint
                refined_path.append(end_point)
                break

        # Move to the next point
        i += 1

    return np.array(refined_path)


def refine_path_reverse(path_points, mesh):
    """
    Refine the path by eliminating unnecessary intermediate points if the direct line
    from a point to the start does not intersect the mesh. Skip rays entirely inside the mesh.
    """
    refined_path = [path_points[-1]]  # Start with the last point
    intersector = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)

    i = len(path_points) - 1
    while i > 0:
        start_point = path_points[0]  # Always check to the first point
        end_point = path_points[i]  # Current end point

        # Subdivide the ray segment to check points along the segment
        num_steps = 5  # Subdivision steps
        segment_points = np.linspace(start_point, end_point, num_steps)

        # Check if all points along the segment lie inside the mesh
        inside_status = mesh.contains(segment_points)
        if np.any(inside_status):  # Entire segment is inside the mesh
            next_point = path_points[i - 1]  # Move to the previous intermediate point
            refined_path.append(next_point)
        else:
            # Perform ray intersection check
            ray_origin = np.array([end_point])  # Shape (1, 3)
            ray_direction = np.array([start_point - end_point])  # Shape (1, 3)
            segment_length = np.linalg.norm(start_point - end_point)  # Length of the segment (scalar)

            # Find intersection locations (exact points)
            locations, index_ray, _ = intersector.intersects_location(
                ray_origins=ray_origin,
                ray_directions=ray_direction
            )

            # Check if any intersection occurs within the segment length
            if locations.shape[0] > 0:  # There are intersections
                distances = np.linalg.norm(locations - end_point, axis=1)  # Calculate distances to intersections

                if np.any(distances < segment_length):  # Intersection before reaching the start point
                    next_point = path_points[i - 1]  # Move to the previous intermediate point
                    refined_path.append(next_point)
                else:
                    # No intersection, add the start point
                    refined_path.append(start_point)
                    break
            else:
                # No intersection, add the start point
                refined_path.append(start_point)
                break

        # Move to the previous point
        i -= 1

    return np.array(refined_path[::-1])  # Reverse the path to maintain the correct order


def refine_path_optimized(path_points, mesh, num_steps=10):
    """
    Refine a path by minimizing points while ensuring the path does not cross the mesh
    or have points inside it.

    Parameters:
        path_points (np.ndarray): Array of points representing the path (N x 3).
        mesh (trimesh.Trimesh): The 3D mesh object.
        num_steps (int): Number of subdivisions for checking segments.

    Returns:
        np.ndarray: Refined path as an array of 3D points.
    """
    refined_path = [path_points[0]]  # Start with the first point
    intersector = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)

    i = 0
    while i < len(path_points) - 1:
        start_point = path_points[i]
        next_index = i + 1  # Default next index

        # Try to connect to farther points
        for j in range(i + 1, len(path_points)):
            end_point = path_points[j]

            # Subdivide the segment and check for points inside the mesh
            segment_points = np.linspace(start_point, end_point, num_steps)
            if np.any(mesh.contains(segment_points)):  # Points inside the mesh
                break

            # Perform ray intersection check
            ray_origin = np.array([start_point])
            ray_direction = np.array([end_point - start_point])
            segment_length = np.linalg.norm(end_point - start_point)

            locations, _, _ = intersector.intersects_location(
                ray_origins=ray_origin,
                ray_directions=ray_direction
            )

            # Check if any intersection is within the segment
            if locations.shape[0] > 0:
                distances = np.linalg.norm(locations - start_point, axis=1)
                if np.any(distances < segment_length):  # Intersection occurs
                    break

            # Update the next valid index if this segment is valid
            next_index = j

        # Add the farthest valid point and update the starting index
        refined_path.append(path_points[next_index])
        i = next_index  # Move to the next point

    # Ensure the final point is included by checking element-wise equality
    if not np.array_equal(refined_path[-1], path_points[-1]):
        refined_path.append(path_points[-1])

    return np.array(refined_path)


def best_refine_path(path_points, mesh, num_steps=5):
    """
    SUPER SLOW, ONLY VALID FOR SHORT PATHS
    Refine the path to the shortest possible segments that do not cross the mesh.

    Parameters:
        path_points (np.ndarray): Array of points representing the path (N x 3).
        mesh (trimesh.Trimesh): The 3D mesh object.
        num_steps (int): Number of subdivisions for checking segments.

    Returns:
        np.ndarray: Refined path as an array of 3D points.
    """
    refined_path = [path_points[0]]  # Start with the first point
    intersector = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)

    i = 0
    while i < len(path_points) - 1:
        found_valid_segment = False  # Flag to track if a valid segment was found

        # Start checking from the last point backward
        for j in range(len(path_points) - 1, i, -1):
            start_point = path_points[i]
            end_point = path_points[j]

            # Subdivide the segment into points and check each for collisions
            segment_points = np.linspace(start_point, end_point, num_steps)
            if np.any(mesh.contains(segment_points)):  # Points inside the mesh
                continue

            # Perform ray intersection check for the entire segment
            ray_origin = np.array([start_point])
            ray_direction = np.array([end_point - start_point])
            segment_length = np.linalg.norm(end_point - start_point)

            locations, _, _ = intersector.intersects_location(
                ray_origins=ray_origin,
                ray_directions=ray_direction
            )

            # If any intersection occurs along the segment, skip this segment
            if locations.shape[0] > 0:
                distances = np.linalg.norm(locations - start_point, axis=1)
                if np.any(distances < segment_length):  # Intersection occurs
                    continue

            # If we reach here, the segment is valid; add the endpoint to the path
            refined_path.append(path_points[j])
            i = j  # Move the starting point forward
            found_valid_segment = True
            break

        # If no valid segment is found, move to the next point (should rarely happen)
        if not found_valid_segment:
            refined_path.append(path_points[i + 1])
            i += 1
    # Ensure the final point is included
    if not np.array_equal(refined_path[-1], path_points[-1]):
        refined_path.append(path_points[-1])

    return np.array(refined_path)


def write_pml_file(path_points, input_pdb, output_dir=None):
    """
    Write a PyMOL script to visualize the path points, color the atoms, and calculate distances
    between consecutive points in the `path_points` list. The script will be saved to the 
    output file `output_name`.
    """

    with open(f'{output_dir}/SASP.pml', 'w') as pml_file:

        abs_path = os.path.abspath(input_pdb)

        pml_file.write(f"load {abs_path}\n")

        pml_file.write("set transparency, 0.5, All\n")
        # pml_file.write("color lightpink, chain A\n")
        # pml_file.write("color palecyan, chain B\n")

        pml_file.write("hide everything, All\n")
        pml_file.write("show surface, All\n")
        pml_file.write("show wire, All\n")
        pml_file.write('util.color_chains("All",_self=cmd)\n')

        # Create pseudoatoms for each point in the path and show them as spheres
        for idx, point in enumerate(path_points):
            pml_file.write(f"pseudoatom {idx + 1}, pos={point.tolist()}\n")
            pml_file.write(f"show spheres, {idx + 1}\n")
            pml_file.write(f"color pink, {idx + 1}\n")

        # Calculate and display distances between all pseudoatoms
        for i in range(len(path_points)):
            pml_file.write(f"distance {i}-{i + 1}, {i}, {i + 1}\n")
            pml_file.write(f"color green, {i}-{i + 1}\n")

        # Add end of the script
        pml_file.write("\n")

        pml_file.write(f"distance eucl,1,{len(path_points)}\n")
        pml_file.write("color red, eucl\n")

        # pml_file.write(f"total green dist: {min_dist}")
    print(f"PML file {output_dir}/SASP.pml has been created.")


def detect_chain_contacts(input_pdb, contact_distance=4.0):
    """
    Detect if chains in a PDB file are in contact (forming a complex) or separate.
    
    Parameters:
        input_pdb (str): Path to the PDB file.
        contact_distance (float): Distance threshold in Angstroms to consider chains in contact.
    
    Returns:
        bool: True if chains are in contact (use main), False if separate (use main_no_contacts).
        int: Number of chains detected.
    """
    # Parse PDB and organize atoms by chain
    chains = {}
    with open(input_pdb, 'r') as pdb_file:
        for line in pdb_file:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                chain_id = line[21]  # Chain ID at column 22 (0-based index 21)
                # Extract coordinates
                try:
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())

                    if chain_id not in chains:
                        chains[chain_id] = []
                    chains[chain_id].append(np.array([x, y, z]))
                except ValueError:
                    continue

    # Convert to numpy arrays
    for chain_id in chains:
        chains[chain_id] = np.array(chains[chain_id])

    num_chains = len(chains)

    # If only one chain, it's a monomer - use main
    if num_chains <= 1:
        print(f"Detected {num_chains} chain(s). Using standard mode (main).")
        return True, num_chains

    # If multiple chains, check for contacts
    print(f"Detected {num_chains} chains. Checking for inter-chain contacts...")

    chain_ids = list(chains.keys())
    for i in range(len(chain_ids)):
        for j in range(i + 1, len(chain_ids)):
            chain_a = chains[chain_ids[i]]
            chain_b = chains[chain_ids[j]]

            # Build KDTree for chain B for efficient distance queries
            kdtree = KDTree(chain_b)

            # Query distances from chain A to chain B
            distances, _ = kdtree.query(chain_a)
            min_distance = np.min(distances)

            if min_distance <= contact_distance:
                print(f"Chains {chain_ids[i]} and {chain_ids[j]} are in contact (min distance: {min_distance:.2f} Å).")
                print("Using standard mode (main) for complex structure.")
                return True, num_chains

    print(f"No inter-chain contacts detected within {contact_distance} Å.")
    print("Using no-contact mode (main_no_contacts) for separate chains.")
    return False, num_chains
