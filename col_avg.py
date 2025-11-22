import numpy as np
from PIL import Image
import svgwrite
import xml.etree.ElementTree as ET
import math

def triangle_area(x1, y1, x2, y2, x3, y3):
    """Calculates the area of a triangle given three points."""
    return 0.5 * abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

def is_inside(px, py, v1, v2, v3):
    """Checks if a point (px, py) is inside a triangle defined by v1, v2, v3
    using the Barycentric Coordinate method (area sum check)."""
    
    x1, y1 = v1
    x2, y2 = v2
    x3, y3 = v3

    
    A = triangle_area(x1, y1, x2, y2, x3, y3)
    
    
    A1 = triangle_area(px, py, x2, y2, x3, y3)
    A2 = triangle_area(x1, y1, px, py, x3, y3)
    A3 = triangle_area(x1, y1, x2, y2, px, py)

    
    return abs(A - (A1 + A2 + A3)) < 0.5

def get_triangles_from_svg(svg_path):
    """Parses the exported SVG to extract triangle coordinates and dimensions."""
    triangles = []
    tree = ET.parse(svg_path)
    root = tree.getroot()
    
    
    svg_namespace = '{http://www.w3.org/2000/svg}'
    
    
    width, height = 0, 0
    viewbox = root.get('viewBox')
    if viewbox:
        _, _, w, h = viewbox.replace(',', ' ').split()
        width, height = float(w), float(h)
    else:
        
        
        w_str = root.get('width', '0')
        h_str = root.get('height', '0')
        
        width = float(''.join(c for c in w_str if c.isdigit() or c == '.'))
        height = float(''.join(c for c in h_str if c.isdigit() or c == '.'))
        
        
        if 'in' in w_str: width *= 72
        if 'in' in h_str: height *= 72

    print(f"Reading SVG file: {svg_path}...")
    print(f"SVG Dimensions (User Units): {width}x{height}")
    
    
    for polygon in root.findall(f'.//{svg_namespace}polygon'):
        points_str = polygon.get('points')
        if points_str:
            
            
            values = points_str.replace(',', ' ').split()
            coords = [float(v) for v in values]
            
            
            
            if len(coords) == 8:
                 coords = coords[:6]
            
            
            if len(coords) == 6:
                triangles.append(coords) 
            
    print(f"Found {len(triangles)} triangles in the SVG.")
    return triangles, width, height

def rgb_to_hex(r, g, b):
    """Converts RGB values to a Hex color string."""
    return f'#{r:02x}{g:02x}{b:02x}'



def process_triangles(svg_input_path, image_path, svg_output_path):
    
    triangle_data, svg_width, svg_height = get_triangles_from_svg(svg_input_path)

    
    img = Image.open(image_path).convert("RGB")
    
    pixels = np.array(img) 
    img_width, img_height = img.size

    
    scale_x = img_width / svg_width
    scale_y = img_height / svg_height
    
    print(f"Image Dimensions: {img_width}x{img_height}")
    print(f"Scale Factors: X={scale_x:.4f}, Y={scale_y:.4f}")

    
    
    dwg = svgwrite.Drawing(svg_output_path, profile='full', size=(svg_width, svg_height))
    
    
    print("Starting color averaging...")

    for i, coords in enumerate(triangle_data):
        
        v1 = (coords[0], coords[1])
        v2 = (coords[2], coords[3])
        v3 = (coords[4], coords[5])
        
        
        sv1 = (v1[0] * scale_x, v1[1] * scale_y)
        sv2 = (v2[0] * scale_x, v2[1] * scale_y)
        sv3 = (v3[0] * scale_x, v3[1] * scale_y)

        
        t_min_x = int(math.floor(min(sv1[0], sv2[0], sv3[0])))
        t_max_x = int(math.ceil(max(sv1[0], sv2[0], sv3[0])))
        t_min_y = int(math.floor(min(sv1[1], sv2[1], sv3[1])))
        t_max_y = int(math.ceil(max(sv1[1], sv2[1], sv3[1])))

        
        t_min_x = max(0, t_min_x)
        t_max_x = min(img_width - 1, t_max_x)
        t_min_y = max(0, t_min_y)
        t_max_y = min(img_height - 1, t_max_y)

        total_r, total_g, total_b = 0, 0, 0
        pixel_count = 0

        
        
        for y in range(t_min_y, t_max_y + 1):
            for x in range(t_min_x, t_max_x + 1):
                
                if is_inside(x + 0.5, y + 0.5, sv1, sv2, sv3): 
                    
                    
                    r, g, b = pixels[y, x]
                    total_r += int(r)
                    total_g += int(g)
                    total_b += int(b)
                    pixel_count += 1
        
        
        if pixel_count > 0:
            avg_r = int(total_r / pixel_count)
            avg_g = int(total_g / pixel_count)
            avg_b = int(total_b / pixel_count)
            hex_color = rgb_to_hex(avg_r, avg_g, avg_b)
        else:
            
            
            
            cx = int((sv1[0] + sv2[0] + sv3[0]) / 3)
            cy = int((sv1[1] + sv2[1] + sv3[1]) / 3)
            if 0 <= cx < img_width and 0 <= cy < img_height:
                 r, g, b = pixels[cy, cx]
                 hex_color = rgb_to_hex(r, g, b)
            else:
                 hex_color = '#000000'
        
        
        dwg.add(dwg.polygon(
            points=[v1, v2, v3],
            fill=hex_color,
            stroke='none' 
        ))
        
        if (i + 1) % 10000 == 0:
            print(f"Processed {i + 1} triangles...")

    
    dwg.save()
    print(f"\nProcessing complete. New SVG saved to {svg_output_path}")


if __name__ == "__main__":
    process_triangles(
        svg_input_path='triangles.svg',    
        image_path='raw.png',   
        svg_output_path='out.svg' 
    )