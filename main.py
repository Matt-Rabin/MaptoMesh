#Matt's little nightmare simulator
#Copyright 2025, all rights reserved.

import numpy as np
import pygame
from numba import njit, prange


class camera:
    def __init__(self, f, theta, phi, d, min_zoom, max_zoom, phi_limit):
        self.theta = theta
        self.phi = phi
        self.d = d
        self.max_zoom = max_zoom 
        self.min_zoom = min_zoom
        self.phi_limit = phi_limit
        self.K = np.diag([f,f,1])

    
    def pan(self, delta_theta, delta_phi):
        self.theta = (self.theta + delta_theta) % (2.0 * np.pi)
        self.phi += delta_phi
        self.phi = np.clip(self.phi, -self.phi_limit, self.phi_limit)

    def zoom(self, delta_d):
        zoom_speed = 0.3
        self.d *= (1 + zoom_speed*delta_d)
        self.d = np.clip(self.d, self.min_zoom, self.max_zoom)

        #This adjusts focal length, just comment it out if you dont like. *= is a real zoom, /= is a dolly zoom
        """if self.d > self.min_zoom and self.d < self.max_zoom:
            f_speed = 0.15
            self.K[0, 0] /= (1 + f_speed * delta_d)
            self.K[1, 1] /= (1 + f_speed * delta_d)"""

    #Converts polar coords to xyz for rendering
    def toXYZ(self):
        x = self.d * np.cos(self.theta) * np.cos(self.phi)
        y = self.d * np.sin(self.theta) * np.cos(self.phi)
        z = self.d * np.sin(self.phi)
        return np.array([x,y,z])
    
    def setK(self,f):
        self.K = np.diag([f,f,1])


class globe_mesh:
    def __init__(self, subdivisions):
        self.subdivisions = subdivisions

        phi = (1 + np.sqrt(5)) / 2.0

        raw_ico = np.array([[ 0.0,  1.0,  phi],
                            [ 0.0, -1.0,  phi],
                            [ 0.0,  1.0, -phi],
                            [ 0.0, -1.0, -phi],
                            [ 1.0,  phi,  0.0],
                            [-1.0,  phi,  0.0],
                            [ 1.0, -phi,  0.0],
                            [-1.0, -phi,  0.0],
                            [ phi,  0.0,  1.0],
                            [ phi,  0.0, -1.0],
                            [-phi,  0.0,  1.0],
                            [-phi,  0.0, -1.0]])
        
        ico_norms = np.linalg.norm(raw_ico, axis=1).reshape(-1, 1)
        self.vertices = raw_ico / ico_norms
        self.vertices = self.vertices[:,[0,2,1]]
        
        #Needs to be ints since we use it to index the vertices
        #Like row 0 indexes vertex 0, 8 and 4,
        #SERIOUSLY DO NOT CHANGE THIS UNLESS MAKING NEW SHAPE - I MADE SURE THAT ALL OF THESE ARE PRESERVING WINDING ORDER
        self.triangles = np.array([[ 0,  8,  4],
                                   [ 0,  4,  5],
                                   [ 0,  5, 10],
                                   [ 0, 10,  1],
                                   [ 0,  1,  8],
                                   [ 4,  8,  9],
                                   [ 5,  4,  2],
                                   [10,  5, 11],
                                   [ 1, 10,  7],
                                   [ 8,  1,  6],
                                   [ 9,  8,  6],
                                   [ 2,  4,  9],
                                   [11,  5,  2],
                                   [ 7, 10, 11],
                                   [ 6,  1,  7],
                                   [ 3,  9,  6],
                                   [ 3,  2,  9],
                                   [ 3,  11, 2],
                                   [ 3,  7, 11],
                                   [ 3,  6,  7],
                                ], dtype=np.int32)
        
        self.subdiv()
        
    def subdiv(self):
        """Subdivision that preserves winding order"""
        for iteration in range(self.subdivisions):
            old_triangles = self.triangles
            old_vertices = self.vertices

            new_vertices = list(old_vertices)
            new_triangles = []

            #Check for duplicate edges (nightmare nightmare nightmare)
            edge_cache = {}

            def get_midpoint_vertex(i, j):
                if i > j:
                    i, j = j, i
                
                key = (i, j)
                
                if key in edge_cache:
                    return edge_cache[key]
                
                midpoint = (new_vertices[i] + new_vertices[j]) / 2.0
                midpoint = midpoint / np.linalg.norm(midpoint)
                
                new_vertices.append(midpoint)
                new_idx = len(new_vertices) - 1
                
                edge_cache[key] = new_idx
                
                return new_idx

            # Subdivide each triangle
            for tri in old_triangles:
                a, b, c = tri
                
                ab = get_midpoint_vertex(a, b)
                bc = get_midpoint_vertex(b, c)
                ca = get_midpoint_vertex(c, a)
                
                new_triangles.append([a,  ab, ca])  # A corner
                new_triangles.append([ab, b,  bc])  # B corner
                new_triangles.append([ca, bc, c])   # C Corner
                new_triangles.append([ab, bc, ca])  # Center

            self.vertices = np.array(new_vertices, dtype=np.float64)
            self.triangles = np.array(new_triangles, dtype=np.int32)


def build_uv(vertices: np.ndarray,
             projection: str = "equirectangular") -> np.ndarray:
    x = vertices[:,0]
    y = vertices[:,1]
    z = vertices[:,2]

    lon = np.arctan2(y, x)
    lat = np.arcsin(np.clip(z, -1.0, 1.0))

    if projection == "equirectangular":
        
        lon_min = -np.pi
        lon_max =  np.pi

        lat_min = -np.pi / 2.0
        lat_max =  np.pi / 2.0

        u = (lon - lon_min) / (lon_max - lon_min)
        v = (lat_max - lat) / (lat_max - lat_min)

        return np.stack([u, v], axis=1)
    
    #Still needs to be added
    elif projection == "mercator":
        return

    raise ValueError(f"Unsupported projection: {projection}")

def fix_uv_seams(vertices, uv, triangles):
    new_vertices = list(vertices)
    new_uv = list(uv)
    new_triangles = []
    
    SEAM_THRESHOLD = 0.5
    
    seam_triangles_fixed = 0
    
    for tri_idx, tri in enumerate(triangles):
        # Get U coordinates
        u_vals = uv[tri, 0]
        u_min = u_vals.min()
        u_max = u_vals.max()
        
        if u_max - u_min > SEAM_THRESHOLD:
            
            seam_triangles_fixed += 1
            
            new_tri = []
            for vert_idx in tri:
                vert_u = uv[vert_idx, 0]
                
                # checks if this vertex is on the left side of the seam
                if vert_u < SEAM_THRESHOLD:
                    new_vertices.append(vertices[vert_idx].copy())
                    new_uv.append([vert_u + 1.0, uv[vert_idx, 1]])
                    new_tri.append(len(new_vertices) - 1)
                else:
                    new_tri.append(vert_idx)
            
            new_triangles.append(new_tri)
        else:
            new_triangles.append(tri.tolist())
    
    return (np.array(new_vertices, dtype=np.float64),
            np.array(new_uv, dtype=np.float64),
            np.array(new_triangles, dtype=np.int32))


@njit
def is_front_facing(x0, y0, x1, y1, x2, y2):
    area2 = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)
    return area2 > 0


@njit
def barycentric_weights(px, py, x0, y0, x1, y1, x2, y2):
    denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    if denom == 0.0:
        return -1.0, -1.0, -1.0
    alpha = ((y1 - y2) * (px - x2) + (x2 - x1) * (py - y2)) / denom
    beta  = ((y2 - y0) * (px - x2) + (x0 - x2) * (py - y2)) / denom
    gamma = 1.0 - alpha - beta
    return alpha, beta, gamma


@njit
def rasterize_triangle(framebuffer, zbuffer, texture,
                       x0, y0, z0, u0, v0,
                       x1, y1, z1, u1, v1,
                       x2, y2, z2, u2, v2,
                       width, height, tex_h, tex_w):
    
    # Back-face culling
    if not is_front_facing(x0, y0, x1, y1, x2, y2):
        return
    
    # Bounding box
    min_x = max(int(min(x0, x1, x2)), 0)
    max_x = min(int(max(x0, x1, x2)), width - 1)
    min_y = max(int(min(y0, y1, y2)), 0)
    max_y = min(int(max(y0, y1, y2)), height - 1)
    
    if min_x > max_x or min_y > max_y:
        return
    
    # Rasterize
    for py_i in range(min_y, max_y + 1):
        for px_i in range(min_x, max_x + 1):
            alpha, beta, gamma = barycentric_weights(
                px_i + 0.5, py_i + 0.5,
                x0, y0, x1, y1, x2, y2
            )
            if alpha < 0 or beta < 0 or gamma < 0:
                continue
            
            # Depth test
            z = alpha * z0 + beta * z1 + gamma * z2
            if z >= zbuffer[py_i, px_i]:
                continue
            
            # UV interpolation
            u_tex = alpha * u0 + beta * u1 + gamma * u2
            v_tex = alpha * v0 + beta * v1 + gamma * v2

            #Fixes seam issue (omfg pain)
            u_tex = u_tex - int(u_tex)
            if u_tex < 0:
                u_tex += 1.0
            
            # Sample texture
            tx = int(min(max(u_tex * (tex_w - 1), 0), tex_w - 1))
            ty = int(min(max(v_tex * (tex_h - 1), 0), tex_h - 1))
            
            # Write pixel
            zbuffer[py_i, px_i] = z
            framebuffer[py_i, px_i, 0] = texture[ty, tx, 0]
            framebuffer[py_i, px_i, 1] = texture[ty, tx, 1]
            framebuffer[py_i, px_i, 2] = texture[ty, tx, 2]


def render(cam, mesh, texture, width, height):
    tex_h, tex_w, _ = texture.shape

    # camera position and view matrix
    cam_origin = cam.toXYZ()
    world_origin = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    forward = world_origin - cam_origin
    forward = forward / np.linalg.norm(forward)

    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    right = np.cross(forward, world_up)
    right = right / np.linalg.norm(right)

    up = np.cross(right, forward)

    R = np.stack([right, up, forward], axis=0)
    t = -R @ cam_origin

    view = np.eye(4, dtype=np.float32)
    view[:3, :3] = R
    view[:3, 3] = t

    # convert vertices into camera space
    ones = np.ones((mesh.vertices.shape[0], 1), dtype=np.float32)
    verts_h = np.hstack([mesh.vertices, ones])
    cam_pts = (view @ verts_h.T).T[:, :3]
    X = cam_pts[:, 0]
    Y = cam_pts[:, 1]
    Z = cam_pts[:, 2]

    # project to screen with K
    f = cam.K[0, 0]
    eps = 1e-6
    Z_safe = np.maximum(Z, eps)

    x_n = X / Z_safe
    y_n = Y / Z_safe

    px = f * x_n
    py = f * y_n

    u_screen = px + width / 2.0
    v_screen = height / 2.0 - py

    # hemisphere cull prep
    cam_origin_dir = cam_origin / np.linalg.norm(cam_origin)
    front_val = mesh.vertices @ cam_origin_dir

    # init buffers
    framebuffer = np.zeros((height, width, 3), dtype=np.float32)
    zbuffer = np.full((height, width), np.inf, dtype=np.float32)

    # Triangle loop
    near = 0.1
    
    for (a, b, c) in mesh.triangles:
        Za, Zb, Zc = Z[a], Z[b], Z[c]
        
        #Behind camera culling
        if Za <= near and Zb <= near and Zc <= near:
            continue
        
        #Hemisphere Culling
        if front_val[a] < -0.1 and front_val[b] < -0.1 and front_val[c] < -0.1:
            continue
        
        # Clip Z values to prevent division issues
        Za = max(Za, near)
        Zb = max(Zb, near)
        Zc = max(Zc, near)
        
        x0, y0 = u_screen[a], v_screen[a]
        x1, y1 = u_screen[b], v_screen[b]
        x2, y2 = u_screen[c], v_screen[c]

        u0, v0 = mesh.uv[a]
        u1, v1 = mesh.uv[b]
        u2, v2 = mesh.uv[c]

        rasterize_triangle(
            framebuffer, zbuffer, texture,
            x0, y0, Za, u0, v0,
            x1, y1, Zb, u1, v1,
            x2, y2, Zc, u2, v2,
            width, height, tex_h, tex_w
        )

    return framebuffer


def handle_input(cam,
                 mouse_dx, mouse_dy,
                 scroll_delta,
                 keys,
                 mouse_sensitivity=0.005,
                 zoom_sensitivity=0.2,
                 key_pan_speed=0.02):
    # Mouse orbit
    if mouse_dx != 0 or mouse_dy != 0:
        dtheta = -mouse_dx * mouse_sensitivity  
        dphi   =  mouse_dy * mouse_sensitivity
        cam.pan(dtheta, dphi)

    # Scroll
    if scroll_delta != 0:
        cam.zoom(-scroll_delta * zoom_sensitivity)

    # Arrow keys
    if 'a' in keys:
        cam.pan(-key_pan_speed, 0.0)
    if 'd' in keys:
        cam.pan(+key_pan_speed, 0.0)
    if 'w' in keys:
        cam.pan(0.0, +key_pan_speed)
    if 's' in keys:
        cam.pan(0.0, -key_pan_speed)
    if 'q' in keys:
        cam.zoom(+key_pan_speed)
    if 'e' in keys:
        cam.zoom(-key_pan_speed)


def main():
    # init window
    pygame.init()
    width, height = 1200, 600
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Globe Viewer")
    clock = pygame.time.Clock()

    # load texture
    earth_surface = pygame.image.load("earth.webp").convert()
    tex_w, tex_h = earth_surface.get_size()
    tex_bytes = pygame.image.tostring(earth_surface, "RGB")
    texture = np.frombuffer(tex_bytes, dtype=np.uint8).reshape((tex_h, tex_w, 3))

    # build globe
    mesh = globe_mesh(subdivisions=3)
    mesh.uv = build_uv(mesh.vertices, projection="equirectangular")

    mesh.vertices, mesh.uv, mesh.triangles = fix_uv_seams(mesh.vertices, mesh.uv, mesh.triangles)

    # camera setup
    cam = camera(f=300.0, theta=0.0, phi=0.4, d=1.5, min_zoom=1.1, max_zoom=3, phi_limit=np.radians(89.0))

    running = True
    last_mouse_pos = None

    # auto orbit state
    auto_orbit = True
    orbit_speed = 0.002

    # FPS tracking
    frame_count = 0
    fps_timer = pygame.time.get_ticks()

    while running:

        mouse_dx = 0.0
        mouse_dy = 0.0
        scroll_delta = 0.0
        keys = set()

        # --- event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif auto_orbit and event.type in (
                pygame.MOUSEBUTTONDOWN,
                pygame.MOUSEWHEEL,
                pygame.KEYDOWN,
            ):
                auto_orbit = False

            elif event.type == pygame.MOUSEMOTION:
                if pygame.mouse.get_pressed()[0]:  # left button drag
                    if last_mouse_pos is not None:
                        x, y = event.pos
                        lx, ly = last_mouse_pos
                        mouse_dx += (x - lx)
                        mouse_dy += (y - ly)
                    last_mouse_pos = event.pos
                else:
                    last_mouse_pos = event.pos

            elif event.type == pygame.MOUSEWHEEL:
                scroll_delta += event.y

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        pressed = pygame.key.get_pressed()
        if pressed[pygame.K_a]:
            keys.add("a")
        if pressed[pygame.K_d]:
            keys.add("d")
        if pressed[pygame.K_w]:
            keys.add("w")
        if pressed[pygame.K_s]:
            keys.add("s")
        if pressed[pygame.K_q]:
            keys.add("q")
        if pressed[pygame.K_e]:
            keys.add("e")

        # --- update camera from inputs ---
        handle_input(cam, mouse_dx, mouse_dy, scroll_delta, keys)

        if auto_orbit:
            cam.pan(orbit_speed, 0.0)

        # --- render frame ---
        framebuffer = render(cam, mesh, texture, width, height)

        # --- blit framebuffer ---
        fb = np.clip(framebuffer, 0, 255).astype(np.uint8)
        surf = pygame.image.frombuffer(fb.tobytes(), (width, height), "RGB")
        screen.blit(surf, (0, 0))
        
        # FPS display
        frame_count += 1
        if pygame.time.get_ticks() - fps_timer > 1000:
            fps = frame_count
            pygame.display.set_caption(f"Globe Viewer (Optimized) - {fps} FPS")
            frame_count = 0
            fps_timer = pygame.time.get_ticks()
        
        pygame.display.flip()

        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()
