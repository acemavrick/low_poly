import numpy as np
from scipy.spatial import Delaunay
from PIL import Image, ImageDraw
import svgwrite
from svgpathtools import svg2paths
import sys
import multiprocessing
import time

# ------------------------------------------------------------------------------
# -- CONFIG
INPUT_SVG = "input.svg"     # input SVG file
REF_IMAGE = "raw.png"        # reference image
OUTPUT_SVG = "delaunay.svg"  # output SVG file

# -- adjust these values to tweak output
SAMPLE_INTERVAL = 30  # sample interval for SVG parsing; less == detailed
RANDOM_POINTS_COUNT = 800 # random points count for triangulation

# when adding random points, we can either:
# - add them uniformly across the image (normal)
# - add them based on the existing/sampled points (density sampling)
USE_DENSITY_SAMPLING = True
DENSITY_DISTRIBUTION_RATIO = 0.7  # % of points added that are near edges
DENSITY_SPREAD = 10.0  # spread of points near edges

# jitter the post-triangulation points to make the output look more human-made
# humans usually don't produce perfect delaunay triangulations
USE_JITTER = True
JITTER_PERCENTAGE = 0.9 # % of points to jitter
JITTER_STRENGTH = 1.5   # max pixel shift
# ------------------------------------------------------------------------------

## setup
global_img_base = None
global_img_shape = None

def init_worker(shared_array, shape):
    global global_img_base
    global global_img_shape
    global_img_base = shared_array
    global_img_shape = shape

def process_triangle_batch(triangle_indices_batch, all_points_svg, scale_factor):
    """
    This function runs on a separate core. It takes a list of triangle indices,
    calculates their colors, and returns the result data.
    """
    results = []
    scale_x, scale_y = scale_factor
    
    # use global shared memory to access the image
    img_view = np.frombuffer(global_img_base, dtype=np.uint8).reshape(global_img_shape)
    height, width, _ = global_img_shape


    for indices in triangle_indices_batch:
        # 1. get svg coords
        tri_svg = all_points_svg[indices]
        
        # 2. scale to image coords
        tri_img = tri_svg * [scale_x, scale_y]
        
        # 3. create mask & average
        # we need a local mask for this specific triangle
        # instead of a full image mask, crop to the triangle's bounding box
        # to save memory/time on mask creation.
        
        min_x = int(np.min(tri_img[:, 0]))
        max_x = int(np.max(tri_img[:, 0])) + 1
        min_y = int(np.min(tri_img[:, 1]))
        max_y = int(np.max(tri_img[:, 1])) + 1
        
        # clamp to image bounds
        min_x = max(0, min_x); max_x = min(width, max_x)
        min_y = max(0, min_y); max_y = min(height, max_y)
        
        w = max_x - min_x
        h = max_y - min_y
        
        if w <= 0 or h <= 0:
            results.append((tri_svg, "#000000"))
            continue

        # extract just the relevant slice of the image
        img_slice = img_view[min_y:max_y, min_x:max_x]
        
        # create mask relative to the slice
        mask_img = Image.new('L', (w, h), 0)
        draw = ImageDraw.Draw(mask_img)
        
        # adjust polygon points to be relative to the slice
        poly_relative = [(p[0] - min_x, p[1] - min_y) for p in tri_img]
        draw.polygon(poly_relative, outline=1, fill=1)
        
        mask = np.array(mask_img).astype(bool)
        
        # average
        try:
            pixels_inside = img_slice[mask]
            if len(pixels_inside) > 0:
                avg = pixels_inside.mean(axis=0)
                hex_color = '#{:02x}{:02x}{:02x}'.format(int(avg[0]), int(avg[1]), int(avg[2]))
            else:
                hex_color = "#000000"
        except Exception:
            hex_color = "#000000"

        results.append((tri_svg, hex_color))
        
    return results

# helpers
def get_points_from_svg(svg_path, sample_interval):
    print(f"parsing {svg_path}...")
    points = []
    paths, attributes = svg2paths(svg_path)
    max_x, max_y = 0, 0
    for path in paths:
        if path.length() == 0: continue
        num_segments = max(1, int(path.length() / sample_interval))
        for i in range(num_segments + 1):
            p = path.point(i / num_segments)
            x, y = p.real, p.imag
            points.append([x, y])
            if x > max_x: max_x = x
            if y > max_y: max_y = y
    return np.array(points), max_x, max_y

def add_random_points(w, h, count, existing):
    print(f"adding {count} random points...")
    corners = np.array([[0,0], [w, 0], [w, h], [0, h]])

    if not USE_DENSITY_SAMPLING:
        new_pts = np.random.rand(count, 2) * [w, h]
        return np.vstack([existing, new_pts, corners])

    # density sampling
    density_count = int(count * DENSITY_DISTRIBUTION_RATIO)
    uniform_count = count - density_count

    # uniform points
    uniform_pts = np.random.rand(uniform_count, 2) * [w, h]

    # density points (sample from existing edge points + noise)
    indices = np.random.choice(len(existing), density_count)
    base_pts = existing[indices]
    noise = np.random.normal(0, DENSITY_SPREAD, base_pts.shape)
    density_pts = base_pts + noise

    # clamp to bounds
    density_pts[:, 0] = np.clip(density_pts[:, 0], 0, w)
    density_pts[:, 1] = np.clip(density_pts[:, 1], 0, h)

    return np.vstack([existing, uniform_pts, density_pts, corners])

def apply_jitter(points, w, h):
    if not USE_JITTER:
        return points

    print("applying jitter...")
    
    jitter_pts = points.copy()
    num_points = len(points) - 4 # exclude corners
    
    # select random subset
    mask = np.random.random(num_points) < JITTER_PERCENTAGE
    indices = np.where(mask)[0]
    
    offsets = (np.random.rand(len(indices), 2) - 0.5) * 2 * JITTER_STRENGTH
    jitter_pts[indices] += offsets
    
    # clamp
    jitter_pts[:, 0] = np.clip(jitter_pts[:, 0], 0, w)
    jitter_pts[:, 1] = np.clip(jitter_pts[:, 1], 0, h)
    
    # restore corners exactly, to be safe
    jitter_pts[-4:] = points[-4:]
    
    return jitter_pts

## main
if __name__ == "__main__":
    start_time = time.time()
    
    # setup data
    print("loading data...")
    try:
        ref_img = Image.open(REF_IMAGE).convert("RGB")
    except:
        print("image not found"); sys.exit()
        
    img_array = np.array(ref_img)
    h, w, c = img_array.shape
    
    outline_points, svg_w, svg_h = get_points_from_svg(INPUT_SVG, SAMPLE_INTERVAL)
    all_points = add_random_points(svg_w, svg_h, RANDOM_POINTS_COUNT, outline_points)
    
    scale_x = w / svg_w
    scale_y = h / svg_h
    print(f"scale: {scale_x:.2f}, {scale_y:.2f}")

    # triangulate
    print("triangulating...")
    tri = Delaunay(all_points)
    
    # apply jitter AFTER triangulation to warp the mesh without breaking topology
    all_points = apply_jitter(all_points, svg_w, svg_h)
    
    triangles = tri.simplices
    print(f"total triangles: {len(triangles)}")

    print("starting parallel processing...")
    num_cores = multiprocessing.cpu_count()
    print(f"using {num_cores} cores.")

    # split triangles into chunks
    chunk_size = len(triangles) // num_cores
    chunks = [triangles[i:i + chunk_size] for i in range(0, len(triangles), chunk_size)]

    # flatten the image array to pass it efficiently if using shared memory
    # though standard argument passing usually works fine unless the image is 4k+
    # create a RawArray in shared memory to avoid copying the image to every core
    # prevents Memory Errors on large images
    shared_img_base = multiprocessing.RawArray('B', h * w * c)
    shared_img_np = np.frombuffer(shared_img_base, dtype=np.uint8).reshape((h, w, c))
    np.copyto(shared_img_np, img_array)

    # init pool with the shared array
    pool = multiprocessing.Pool(processes=num_cores, initializer=init_worker, initargs=(shared_img_base, (h, w, c)))
    
    # prepare arguments for each chunk
    func_args = []
    for chunk in chunks:
        func_args.append((chunk, all_points, (scale_x, scale_y)))

    # run the pool
    results_nested = pool.starmap(process_triangle_batch, func_args)
    
    pool.close()
    pool.join()

    # write output
    print("writing SVG...")
    dwg = svgwrite.Drawing(OUTPUT_SVG, profile='full', size=(svg_w, svg_h))
    
    count = 0
    # flatten results
    for batch_result in results_nested:
        for (tri_coords, hex_color) in batch_result:
            points_list = [tuple(pt) for pt in tri_coords]
            dwg.add(dwg.polygon(points=points_list, fill=hex_color, stroke='none'))
            count += 1

    dwg.save()
    print(f"done! processed {count} triangles in {time.time() - start_time:.2f} seconds.")