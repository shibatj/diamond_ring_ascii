#!/usr/bin/env python3
"""
ASCII Diamond Ring - a 3D ring drawn only with letters, numbers and symbols (monochrome).

Pipeline (every frame):
  1. Model    point-sampled 3D surfaces: band (+ milgrain, pave), basket, prongs, halo,
              and a 57-facet round brilliant built as a real polygon mesh.
  2. Forming  a faint dotted "ghost" of the ring fades in; two points of light run up the
              band from the bottom; the setting rises; the stone's facet edges are drawn
              as line art, then its facets fill in; a burst of light completes it.
  3. Project  rotate -> perspective divide -> z-buffer into a character grid.
  4. Light    metal  : Blinn-Phong + rim + sky reflection.
              diamond: Fresnel reflection + fake refraction / internal bounce into a studio
                       environment -> the black/white "scintillation" of a real stone.
  5. ASCII    brightness -> character ramp; silhouettes and facet edges -> / \\ | _ -
  6. Extras   light rays, orbit trails, sparks, glowing motes, floor reflection, bloom.

Controls : ESC quit | SPACE pause | R restart | B bloom | H hud | S screenshot
Run      : python diamond_ring.py              (fullscreen, native resolution)
           python diamond_ring.py --windowed   (1080x960 window)
Offline  : python diamond_ring.py --headless --snap 3,7,12 --out shots
           python diamond_ring.py --headless --record 20 --out frames
"""

import argparse
import math
import os
import sys

import numpy as np

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame  # noqa: E402

TAU = 2.0 * math.pi

# ----------------------------------------------------------------------------- timeline
GHOST_IN = 1.2                       # s : dotted ghost of the ring fades in
FORM_START = 1.0                     # s : the points of light start drawing
FORM_DUR = 7.5                       # s : time to build the whole ring
FORM_END = FORM_START + FORM_DUR
ROT_START = FORM_END + 1.2           # s : ring starts turning
ROT_SPEED = 24.0                     # deg / s

# how the reveal parameter s (0..1) is spent
S_BAND = 0.55                        # band (two lights, bottom -> top)
S_SET = 0.72                         # basket, prongs, halo (rising)
S_EDGE = (0.72, 0.86)                # stone: facet edges drawn as line art
S_FILL = (0.80, 0.97)                # stone: facets fill in
DISSOLVE = 0.012                     # width of the soft "dissolve" front

# ----------------------------------------------------------------------------- characters
METAL_RAMP = " .:;-=+<>il!jt?)(}{][rcvunxzoeasJCLYUQ0OZ$%&8B@"
DIAM_RAMP = " .:-=+*xoO#%$&8@"
REFL_RAMP = " .,:;i1jt"
GHOST_CH = np.frombuffer(b".:'`,", np.uint8)
SPARK_CH = np.array([ord(c) for c in ".'+*"], np.uint8)
TRAIL_CH = np.array([ord(c) for c in "&88%%++::....."], np.uint8)
DUST_CH = [ord(c) for c in ".'`,:+&%8*"]

# ----------------------------------------------------------------------------- colours
BG = (0, 0, 0)
METAL, DIAMOND, SPARK, RAY, DUST, REFLECT, GLINT, GLINT2, ORBIT, GHOST = range(10)
PALETTES = [                         # strictly monochrome: neutral greys only
    (232, 232, 232),  # METAL
    (255, 255, 255),  # DIAMOND
    (255, 255, 255),  # SPARK
    (215, 215, 215),  # RAY
    (150, 150, 150),  # DUST
    (150, 150, 150),  # REFLECT
    (255, 255, 255),  # GLINT
    (255, 255, 255),  # GLINT2
    (205, 205, 205),  # ORBIT
    (125, 125, 125),  # GHOST
]
HUD_GREY = (140, 140, 140)
HUD_HI = (215, 215, 215)


def norm(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def ease(x):
    """Gentle ease-in-out that never stalls (keeps 65% linear speed at the ends)."""
    return 0.65 * x + 0.35 * (0.5 - 0.5 * math.cos(math.pi * x))


def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def dir_codes(dx, dy):
    """Char that best draws a line with screen direction (dx right, dy down)."""
    a = np.degrees(np.arctan2(-np.asarray(dy, float), np.asarray(dx, float))) % 180.0
    out = np.full(np.shape(a), ord("-"), np.uint8)
    out[(a >= 22.5) & (a < 67.5)] = ord("/")
    out[(a >= 67.5) & (a < 112.5)] = ord("|")
    out[(a >= 112.5) & (a < 157.5)] = ord("\\")
    return out


def s_band_of(u):
    """Reveal time of band angle u: two lights start at the bottom and run up both sides."""
    d = np.abs(((np.asarray(u) + math.pi / 2 + math.pi) % TAU) - math.pi)
    return S_BAND * d / math.pi


BELT_CH = np.frombuffer(b".,:;+*", np.uint8)

# ============================================================================ geometry
def band_profile(u):
    """Semi-thickness (radial) and semi-width (axial) of the band at angle u."""
    shoulder = np.clip(np.sin(u), 0.0, None) ** 4         # 1 under the setting
    return 0.125 + 0.045 * shoulder, 0.17 + 0.05 * shoulder


def band_geometry(R, nu=700, nv=76):
    """Ring shank: torus around the z axis with an elliptical, top-thickened profile."""
    u = np.linspace(0.0, TAU, nu, endpoint=False)
    v = np.linspace(0.0, TAU, nv, endpoint=False)
    U, V = [x.ravel() for x in np.meshgrid(u, v, indexing="ij")]
    t, w = band_profile(U)
    er = np.stack([np.cos(U), np.sin(U), np.zeros_like(U)], 1)
    ez = np.array([0.0, 0.0, 1.0])
    P = er * (R + t * np.cos(V))[:, None] + ez * (w * np.sin(V))[:, None]
    # normal of the ellipse (x/t)^2 + (z/w)^2 = 1  ->  (cos v / t, sin v / w)
    N = norm(er * (np.cos(V) / t)[:, None] + ez * (np.sin(V) / w)[:, None])
    return P, N, U


def band_frame(R, u, v):
    """Point + normal on the band surface at (u, v)."""
    t, w = band_profile(u)
    er = np.array([math.cos(u), math.sin(u), 0.0])
    ez = np.array([0.0, 0.0, 1.0])
    p = er * (R + t * math.cos(v)) + ez * (w * math.sin(v))
    n = norm(er * (math.cos(v) / t) + ez * (math.sin(v) / w))
    return p, n


def tube_geometry(points, radius, n_around=16, step=0.006):
    """Thin cylinder following a polyline (prongs)."""
    pts = np.asarray(points, float)
    Ps, Ns = [], []
    for a, b in zip(pts[:-1], pts[1:]):
        d = b - a
        L = float(np.linalg.norm(d))
        d = d / L
        helper = np.array([0.0, 1.0, 0.0]) if abs(d[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
        e1 = norm(np.cross(d, helper))
        e2 = np.cross(d, e1)
        k = np.linspace(0.0, 1.0, max(4, int(L / step)))
        th = np.linspace(0.0, TAU, n_around, endpoint=False)
        K, TH = [x.ravel() for x in np.meshgrid(k, th, indexing="ij")]
        nrm = np.cos(TH)[:, None] * e1 + np.sin(TH)[:, None] * e2
        Ps.append(a + (K * L)[:, None] * d + radius * nrm)
        Ns.append(nrm)
    return np.vstack(Ps), np.vstack(Ns)


def sphere_geometry(c, r, n=160):
    """Fibonacci-sphere bead."""
    i = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    th = math.pi * (1.0 + 5 ** 0.5) * i
    N = np.stack([np.cos(th) * np.sin(phi), np.cos(phi), np.sin(th) * np.sin(phi)], 1)
    return np.asarray(c) + r * N, N


def faceted_bead(c, r, n=120):
    """Tiny stone / milgrain bead: normals snapped to a coarse lattice -> a few flat facets."""
    P, N = sphere_geometry(c, r, n)
    return P, norm(np.round(N * 1.6) / 1.6)


def ytorus_geometry(yc, Rm, r, nu=360, nv=16):
    """Small torus around the y axis (basket / halo rails)."""
    u = np.linspace(0.0, TAU, nu, endpoint=False)
    v = np.linspace(0.0, TAU, nv, endpoint=False)
    U, V = [x.ravel() for x in np.meshgrid(u, v, indexing="ij")]
    er = np.stack([np.cos(U), np.zeros_like(U), np.sin(U)], 1)
    ey = np.array([0.0, 1.0, 0.0])
    P = er * (Rm + r * np.cos(V))[:, None] + ey * (yc + r * np.sin(V))[:, None]
    N = er * np.cos(V)[:, None] + ey * np.sin(V)[:, None]
    return P, N


def brilliant_mesh(rg, table=0.56, crown_deg=34.5, pav_deg=40.75,
                   girdle=0.03, star=0.5, lower=0.78):
    """Round brilliant cut as a polygon mesh (axis = y, girdle centred at y = 0).

    Crown : table (1) + stars (8) + kites/bezels (8) + upper girdle (16)
    Girdle: 16 narrow vertical facets
    Pavil.: lower girdle (16) + pavilion mains (8), meeting at the culet
    The star / lower-girdle points are placed so that every kite is exactly planar."""
    rt, g = table * rg, girdle * rg
    yt = g + (rg - rt) * math.tan(math.radians(crown_deg))     # table height
    yc = -g - rg * math.tan(math.radians(pav_deg))              # culet depth
    V = []

    def vert(r, a, y):
        V.append((r * math.cos(a), y, r * math.sin(a)))
        return len(V) - 1

    A8 = [k * TAU / 8 for k in range(8)]
    A16 = [j * TAU / 16 for j in range(16)]
    c225 = math.cos(TAU / 16)
    T = [vert(rt, a, yt) for a in A8]
    GT = [vert(rg, a, g) for a in A16]
    GB = [vert(rg, a, -g) for a in A16]
    # star tips: radius rs at angle (k+.5)*45deg; height chosen so kite k is planar
    rs = rt + (rg - rt) * star
    ys = yt + (rs * c225 - rt) * (g - yt) / (rg - rt)
    S = [vert(rs, a + TAU / 16, ys) for a in A8]
    # lower-girdle points: radius rl; height chosen so pavilion main k is planar
    rl = rg * (1.0 - lower)
    yl = -g + (1.0 - rl * c225 / rg) * (yc + g)
    L = [vert(rl, a + TAU / 16, yl) for a in A8]
    C = vert(0.0, 0.0, yc)

    polys = [T[:]]                                                # table
    for k in range(8):
        k1, km = (k + 1) % 8, (k - 1) % 8
        polys.append([T[k], T[k1], S[k]])                         # star
        polys.append([T[k], S[km], GT[2 * k], S[k]])              # kite / bezel
        polys.append([S[k], GT[2 * k], GT[2 * k + 1]])            # upper girdle x2
        polys.append([S[k], GT[2 * k + 1], GT[(2 * k + 2) % 16]])
    for j in range(16):
        j1 = (j + 1) % 16
        polys.append([GT[j], GT[j1], GB[j1], GB[j]])              # girdle
    for k in range(8):
        km = (k - 1) % 8
        polys.append([L[k], GB[2 * k], GB[2 * k + 1]])            # lower girdle x2
        polys.append([L[k], GB[2 * k + 1], GB[(2 * k + 2) % 16]])
        polys.append([GB[2 * k], L[k], C, L[km]])                 # pavilion main
    return np.array(V), polys, yt, yc


def sample_polygons(V, polys, density, seed=3):
    """Fill every polygon with points (fan-triangulated, area-proportional random points).
    Each polygon gets one outward Newell normal and its own facet id."""
    rng = np.random.default_rng(seed)
    Ps, Ns, F = [], [], []
    for fi, poly in enumerate(polys):
        pts = V[poly]
        n = np.zeros(3)
        for a, b in zip(pts, np.roll(pts, -1, 0)):
            n += [(a[1] - b[1]) * (a[2] + b[2]), (a[2] - b[2]) * (a[0] + b[0]),
                  (a[0] - b[0]) * (a[1] + b[1])]
        n /= np.linalg.norm(n)
        if np.dot(n, pts.mean(0)) < 0:
            n = -n
        for i in range(1, len(pts) - 1):
            A, B, Cc = pts[0], pts[i], pts[i + 1]
            # uniform random points, count proportional to AREA (a grid would oversample
            # long thin triangles); 2 points per density^2 keeps every cell covered
            area = 0.5 * np.linalg.norm(np.cross(B - A, Cc - A))
            k = max(4, int(2.0 * area / density ** 2))
            u, w = rng.random(k), rng.random(k)
            flip = u + w > 1                               # fold the square onto the triangle
            u[flip], w[flip] = 1 - u[flip], 1 - w[flip]
            P = A + (B - A) * u[:, None] + (Cc - A) * w[:, None]
            P = np.vstack([P, A, B, Cc])                   # keep the sharp corners
            Ps.append(P)
            Ns.append(np.repeat(n[None], len(P), 0))
            F.append(np.full(len(P), fi))
    return np.vstack(Ps), np.vstack(Ns), np.concatenate(F)


def polygon_edges(polys):
    edges = set()
    for poly in polys:
        for a, b in zip(poly, poly[1:] + poly[:1]):
            edges.add((min(a, b), max(a, b)))
    return np.array(sorted(edges))


class RingModel:
    R = 1.0                                   # band radius
    YG, RG = 1.66, 0.44                       # stone: girdle height, girdle radius

    def __init__(self, seed=7):
        rng = np.random.default_rng(seed)
        Ps, Ns, Ss, Ms, Phs = [], [], [], [], []

        def add(P, N, s, mat, phase=None):
            Ps.append(P)
            Ns.append(N)
            Ss.append(np.broadcast_to(s, len(P)).astype(float))
            Ms.append(np.full(len(P), mat, np.uint8))
            Phs.append(np.zeros(len(P)) if phase is None else np.broadcast_to(phase, len(P)))

        def s_band(u):
            """Two lights start at the bottom and run up both sides to the top."""
            d = np.abs(((np.asarray(u) + math.pi / 2 + math.pi) % TAU) - math.pi)
            return S_BAND * d / math.pi

        # stone geometry first (its height sets the prong profile)
        Vst, polys, yt, yc = brilliant_mesh(self.RG)
        self.CROWN = yt

        # 1) band + milgrain + pave
        P, N, U = band_geometry(self.R)
        add(P, N, s_band(U), METAL)
        for v in (math.radians(42), math.radians(138), math.radians(-42), math.radians(-138)):
            for u in np.arange(0.0, TAU, 0.045):
                p, n = band_frame(self.R, u, v)
                P, N = faceted_bead(p + n * 0.006, 0.017, 36)
                add(P, N, s_band(u), METAL)
        for side in (math.radians(90), math.radians(-90)):
            for u in np.arange(math.radians(18), math.radians(162), 0.105):
                if abs(u - math.pi / 2) < math.radians(13):
                    continue                  # leave room under the head
                p, n = band_frame(self.R, u, side)
                P, N = faceted_bead(p + n * 0.012, 0.042, 110)
                add(P, N, s_band(u), DIAMOND, rng.random())

        # 2) setting: rises bottom -> top during s in [S_BAND, S_SET]
        y_lo, y_hi = 1.05, self.YG + yt

        def s_set(P):
            return S_BAND + (S_SET - S_BAND) * np.clip((P[:, 1] - y_lo) / (y_hi - y_lo), 0, 1)

        for k in range(6):
            a = math.radians(30 + 60 * k)
            prof = [(0.12, 1.10), (0.30, 1.36), (self.RG + 0.03, self.YG),
                    (self.RG - 0.05, self.YG + yt * 0.6)]
            pts = [(r * math.cos(a), y, r * math.sin(a)) for r, y in prof]
            P, N = tube_geometry(pts, 0.026)
            add(P, N, s_set(P), METAL)
            P, N = sphere_geometry(pts[-1], 0.042, 260)
            add(P, N, s_set(P), METAL)
        for ycen, rr in [(1.36, 0.30), (1.18, 0.175)]:
            P, N = ytorus_geometry(ycen, rr, 0.024)
            add(P, N, s_set(P), METAL)

        # halo: rail + 18 small brilliants around the centre stone
        rh, yh = self.RG + 0.085, self.YG - 0.03
        P, N = ytorus_geometry(yh - 0.045, rh, 0.028, nu=420)
        add(P, N, s_set(P), METAL)
        Vh, ph_polys, _, _ = brilliant_mesh(0.062)
        Ph0, Nh0, fh = sample_polygons(Vh, ph_polys, 0.011)
        for i in range(18):
            a = TAU * i / 18
            P = Ph0 + np.array([rh * math.cos(a), yh, rh * math.sin(a)])
            add(P, Nh0, s_set(P), DIAMOND, rng.random(fh.max() + 1)[fh])

        # 3) centre stone: facets fill in (bottom -> top, slightly shuffled)
        P, N, fid = sample_polygons(Vst, polys, 0.0095)
        cy_f = np.array([Vst[p][:, 1].mean() for p in polys])
        yn_f = (cy_f - yc) / (yt - yc)
        s_f = S_FILL[0] + (S_FILL[1] - S_FILL[0]) * (0.65 * yn_f + 0.35 * rng.random(len(polys)))
        add(P + [0.0, self.YG, 0.0], N, s_f[fid], DIAMOND, rng.random(len(polys))[fid])

        # its facet edges (line art), drawn bottom -> top before the facets fill
        E = polygon_edges(polys)
        ea, eb = Vst[E[:, 0]] + [0, self.YG, 0], Vst[E[:, 1]] + [0, self.YG, 0]
        yn_e = ((ea[:, 1] + eb[:, 1]) / 2 - self.YG - yc) / (yt - yc)
        span = S_EDGE[1] - S_EDGE[0] - 0.035
        self.edge_s = S_EDGE[0] + span * (0.85 * yn_e + 0.15 * rng.random(len(E)))

        self.P = np.vstack(Ps)
        self.N = np.vstack(Ns)
        self.s = np.concatenate(Ss)
        self.mat = np.concatenate(Ms)
        self.phase = np.concatenate(Phs)
        self.jit = rng.random(len(self.P))                 # dissolve / ghost noise

        # centre the model vertically around the origin (= the axis it turns on)
        top, bottom = self.YG + yt, -(self.R + 0.125)
        self.cy = (top + bottom) / 2
        shift = np.array([0.0, self.cy, 0.0])
        self.P -= shift
        self.edge_a, self.edge_b = ea - shift, eb - shift
        self.P32 = self.P.astype(np.float32)               # float32 copies for the hot path
        self.N32 = self.N.astype(np.float32)
        self.diamond_c = np.array([0.0, self.YG, 0.0]) - shift
        self.bottom_y = bottom - self.cy


# ============================================================================ particles
class Sparks:
    """Sparks in screen space with gravity and drag."""
    G = 320.0

    def __init__(self, rng):
        self.rng = rng
        self.clear()

    def clear(self):
        z = np.zeros(0)
        self.x, self.y, self.vx, self.vy, self.life, self.max = z, z, z, z, z, z

    def emit(self, x, y, n, vmin=30.0, vmax=200.0, up=60.0):
        if n <= 0:
            return
        r = self.rng
        a = r.uniform(0, TAU, n)
        v = r.uniform(vmin, vmax, n)
        life = r.uniform(0.5, 1.6, n)
        self.x = np.concatenate([self.x, np.broadcast_to(x, n) + r.normal(0, 2, n)])
        self.y = np.concatenate([self.y, np.broadcast_to(y, n) + r.normal(0, 2, n)])
        self.vx = np.concatenate([self.vx, v * np.cos(a)])
        self.vy = np.concatenate([self.vy, v * np.sin(a) - up])
        self.life = np.concatenate([self.life, life])
        self.max = np.concatenate([self.max, life])

    def step(self, dt):
        if len(self.x) == 0:
            return
        drag = math.exp(-1.2 * dt)
        self.vy = (self.vy + self.G * dt) * drag
        self.vx = self.vx * drag
        self.x = self.x + self.vx * dt
        self.y = self.y + self.vy * dt
        self.life = self.life - dt
        keep = self.life > 0
        for k in ("x", "y", "vx", "vy", "life", "max"):
            setattr(self, k, getattr(self, k)[keep])


# ============================================================================ scene
class Scene:
    GLOW_R = [2.5, 4, 6, 9, 13, 19, 28, 40]                 # gaussian radii (px)
    FLARE_L = [9, 15, 24, 36, 52]                           # star-flare arm lengths (px)

    def __init__(self, W, H, font_size, bloom=True, seed=7):
        self.W, self.H = W, H
        fonts = "consolas,dejavusansmono,menlo,couriernew"
        self.font = pygame.font.SysFont(fonts, font_size)
        self.hud_font = pygame.font.SysFont(fonts, max(13, H // 64))
        self.cw = self.font.size("M")[0]
        self.ch = self.font.get_linesize()
        self.cols, self.rows = W // self.cw, H // self.ch
        self.n = self.cols * self.rows
        self.pos = [((i % self.cols) * self.cw, (i // self.cols) * self.ch) for i in range(self.n)]
        self.glyphs = {}
        self.glow_cache = {}
        self.glows = []
        self.bloom = bloom
        self.hud = True
        self.rng = r = np.random.default_rng(seed)

        self.model = RingModel(seed)
        self.D = 6.0                                       # camera distance
        self.f = (0.66 * H / 2.95) * self.D                # focal length in pixels
        self.cx, self.cy = W / 2, H * 0.45
        self.sparks = Sparks(r)
        k = H / 960.0                                      # pixel-size scale

        # studio environment the diamond reflects / refracts: (direction, sharpness, power)
        self.softboxes = [(norm([-0.55, 0.70, 0.45]), 50, 1.2),
                          (norm([0.60, 0.45, 0.65]), 70, 1.0),
                          (norm([0.0, 1.0, 0.15]), 18, 0.6),
                          (norm([0.0, -0.25, 1.0]), 120, 0.9),
                          (norm([-0.9, -0.1, -0.4]), 40, 0.7)]

        # light rays from the stone (mostly an upward fan + a few strays)
        up = r.uniform(math.radians(55), math.radians(125), 11)
        stray = r.uniform(0, TAU, 6)
        self.ray_ang = np.concatenate([up, stray, [math.radians(270)]])
        self.ray_len = k * np.concatenate([r.uniform(70, 200, 11), r.uniform(40, 120, 6), [330]])
        self.ray_int = np.concatenate([r.uniform(0.25, 0.6, 11), r.uniform(0.15, 0.35, 6), [0.25]])
        self.ray_spd = r.uniform(0.8, 2.5, len(self.ray_ang))
        self.ray_ph = r.uniform(0, TAU, len(self.ray_ang))

        # orbits: (axis-1, axis-2, radius, angular speed, particles)
        d = math.radians(35)
        self.orbits = [
            (np.array([math.cos(d), math.sin(d), 0.0]), np.array([0.0, 0.0, 1.0]), 2.05, 0.9, 4),
            (np.array([-math.cos(d), math.sin(d), 0.0]), np.array([0.0, 0.0, 1.0]), 2.05, -0.75, 4),
            (np.array([1.0, 0.0, 0.0]), norm([0.0, 0.16, 1.0]), 1.65, 0.55, 5),
        ]

        # dust: drifting background glyphs (density independent of resolution)
        nd = int(700 * W * H / (1080 * 960))
        self.dust_x = r.uniform(0, W, nd)
        self.dust_y = r.uniform(0, H, nd)
        self.dust_v = r.uniform(3, 14, nd) * k
        self.dust_ch = r.choice(DUST_CH, nd).astype(np.uint8)
        self.dust_ph = r.uniform(0, TAU, nd)
        self.dust_w = r.uniform(0.4, 2.0, nd)
        self.dust_pal = np.where(r.random(nd) < 0.75, DUST, REFLECT).astype(np.uint8)

        # glowing motes: 3D particles drifting around the ring
        nm = 90
        self.mote_r = r.uniform(1.2, 3.2, nm)
        self.mote_a = r.uniform(0, TAU, nm)
        self.mote_y = r.uniform(-1.8, 2.0, nm)
        self.mote_w = r.uniform(0.05, 0.25, nm) * r.choice([-1, 1], nm)
        self.mote_bob = r.uniform(0.02, 0.12, nm)
        self.mote_ph = r.uniform(0, TAU, nm)
        self.mote_tw = r.uniform(0.6, 2.4, nm)

        # ring-shaped particle regions -------------------------------------------
        # (a) band stream: particles spiralling around the band, inside a torus that
        #     hugs the metal (distance 0.24..0.46 from the band's centre line)
        nb = 1600
        self.bs_u = r.uniform(0, TAU, nb)                  # position around the ring
        self.bs_v = r.uniform(0, TAU, nb)                  # position around the tube
        self.bs_rho = r.uniform(0.24, 0.46, nb)            # distance from centre line
        self.bs_spd = r.uniform(0.25, 0.65, nb)            # flow speed along the band
        self.bs_vspd = r.uniform(0.6, 1.8, nb) * r.choice([-1, 1], nb)   # helical spin
        self.bs_ph = r.uniform(0, TAU, nb)
        self.bs_tw = r.uniform(1.0, 4.0, nb)
        # (b) orbital belt: a wide flat annulus around the whole ring, tilted so its
        #     front passes below / in front and its back above / behind the ring
        nr = 2600
        self.belt_r = np.sqrt(r.uniform(1.55 ** 2, 2.55 ** 2, nr))   # uniform over area
        self.belt_a = r.uniform(0, TAU, nr)
        self.belt_y = r.normal(0, 0.03, nr) * (self.belt_r / 1.55)   # slightly thicker outside
        self.belt_w = 0.55 * (1.55 / self.belt_r) ** 1.5            # Kepler: inner = faster
        self.belt_ph = r.uniform(0, TAU, nr)
        self.belt_tw = r.uniform(0.8, 3.0, nr)
        self.belt_T = rot_z(math.radians(-10)) @ rot_x(math.radians(14))

        self.stats = dict(rotation=0.0, glints=0, letters=0, particles=0)
        self.reset_layers()

    # ------------------------------------------------------------------ glow sprites
    def glow_sprite(self, kind, size, level):
        """kind 0: soft round glow, kind 1: 4-point star flare. level 1..8 = intensity."""
        key = (kind, size, level)
        s = self.glow_cache.get(key)
        if s is not None:
            return s
        peak = 255.0 * level / 8.0
        sc = self.H / 960.0
        if kind == 0:
            rad = self.GLOW_R[size] * sc
            h = int(math.ceil(rad * 2.6))
            yy, xx = np.mgrid[-h:h + 1, -h:h + 1].astype(np.float64)
            q = (xx * xx + yy * yy) / (rad * rad)
            I = peak * (0.55 * np.exp(-q * 3.0) + 0.45 * np.exp(-q * 0.8))
        else:
            L = self.FLARE_L[size] * sc
            h = int(L)
            yy, xx = np.mgrid[-h:h + 1, -h:h + 1].astype(np.float64)

            def arm(a, b, w, ln):
                return np.exp(-(b / w) ** 2) * np.exp(-np.abs(a) / (L * ln))
            u, v = (xx + yy) / math.sqrt(2), (xx - yy) / math.sqrt(2)
            core = np.exp(-(xx * xx + yy * yy) / (L * 0.09) ** 2)
            I = peak * np.clip(arm(xx, yy, 0.9, 0.28) + arm(yy, xx, 0.9, 0.28)
                               + 0.45 * (arm(u, v, 0.8, 0.12) + arm(v, u, 0.8, 0.12)) + core, 0, 1)
        I = np.clip(I, 0, 255).astype(np.uint8)
        s = pygame.surfarray.make_surface(np.repeat(I.T[:, :, None], 3, 2)).convert()
        self.glow_cache[key] = s
        return s

    def glow(self, x, y, size, k, kind=0):
        """Queue a glow at pixel (x, y); k in 0..1 is intensity."""
        lvl = int(round(float(np.clip(k, 0, 1)) * 8))
        if lvl > 0:
            self.glows.append((float(x), float(y), kind, int(size), lvl))

    # ------------------------------------------------------------------ helpers
    def reset_layers(self):
        self.CH = np.full(self.n, 32, np.uint8)
        self.LV = np.zeros(self.n, np.float32)
        self.PAL = np.zeros(self.n, np.uint8)

    def project(self, P):
        depth = self.D - P[:, 2]
        return self.cx + self.f * P[:, 0] / depth, self.cy - self.f * P[:, 1] / depth, depth

    def put(self, x, y, code, lv, pal, under=False, depth=None, zgrid=None):
        """Write glyphs at pixel positions; brighter wins inside a cell.
        under=True   -> only into empty cells
        depth+zgrid  -> only where nearer than the ring (z-test)"""
        x = np.atleast_1d(np.asarray(x, float))
        y = np.atleast_1d(np.asarray(y, float))
        c = np.floor(x / self.cw).astype(np.int64)
        r = np.floor(y / self.ch).astype(np.int64)
        lv = np.broadcast_to(np.asarray(lv, np.float32), x.shape)
        code = np.broadcast_to(np.asarray(code, np.uint8), x.shape)
        pal = np.broadcast_to(np.asarray(pal, np.uint8), x.shape)
        ok = (c >= 0) & (c < self.cols) & (r >= 0) & (r < self.rows) & (lv > 0.015)
        idx = r * self.cols + c
        if depth is not None:
            ok[ok] &= np.broadcast_to(depth, x.shape)[ok] < zgrid[idx[ok]]
        if under:
            ok[ok] &= self.CH[idx[ok]] == 32
        if not ok.any():
            return
        idx, code, lv, pal = idx[ok], code[ok], lv[ok], pal[ok]
        o = np.argsort(lv, kind="stable")                  # brightest written last
        self.CH[idx[o]] = code[o]
        self.LV[idx[o]] = lv[o]
        self.PAL[idx[o]] = pal[o]

    def visible(self, x, y, depth, zbuf):
        c, r = int(x // self.cw), int(y // self.ch)
        if not (0 <= c < self.cols and 0 <= r < self.rows):
            return False
        return depth < zbuf[r * self.cols + c]

    def star(self, x, y, k):
        """Point of light: a small 8-way burst of line glyphs."""
        for a in np.arange(8) * (TAU / 8):
            d = np.arange(1, 5) * self.cw * 0.9
            dx, dy = math.cos(a), -math.sin(a)
            self.put(x + d * dx, y + d * dy * 0.8, dir_codes(dx, dy),
                     k * (1 - d / (5 * self.cw)), RAY)
        self.put(x, y, ord("@"), k, DIAMOND)

    def particle_rings(self, t, M, cur, zbuf):
        """Draw the band stream and the orbital belt, each with a 3-sample motion trail."""
        m = self.model
        ba = float(smoothstep(0.5, 3.0, t))                 # belt fades in early
        xs, ys, ds, ls = [], [], [], []
        for lag, fade in ((0.0, 1.0), (0.07, 0.5), (0.14, 0.25)):
            tt = t - lag
            # (a) band stream - in model space, so it turns with the ring
            u = self.bs_u + tt * self.bs_spd
            v = self.bs_v + tt * self.bs_vspd
            rr = m.R + self.bs_rho * np.cos(v)
            P = np.stack([rr * np.cos(u), rr * np.sin(u) - m.cy,
                          self.bs_rho * np.sin(v)], 1) @ M.T
            built = smoothstep(0.0, 0.05, cur - s_band_of(u))  # only where the band exists
            tw = 0.5 + 0.5 * np.sin(tt * self.bs_tw + self.bs_ph)
            lv = fade * built * (0.12 + 0.42 * tw ** 2) * (0.6 + 0.4 * (P[:, 2] > 0))
            x, y, d = self.project(P)
            xs.append(x); ys.append(y); ds.append(d); ls.append(lv)
            # (b) orbital belt - its own tilted plane
            a = self.belt_a + tt * self.belt_w
            P = np.stack([self.belt_r * np.cos(a), self.belt_y,
                          self.belt_r * np.sin(a)], 1) @ self.belt_T.T
            tw = 0.5 + 0.5 * np.sin(tt * self.belt_tw + self.belt_ph)
            lv = fade * ba * (0.10 + 0.42 * tw ** 2) * (0.55 + 0.45 * (P[:, 2] > 0))
            x, y, d = self.project(P)
            xs.append(x); ys.append(y); ds.append(d); ls.append(lv)
        x, y, d, lv = (np.concatenate(a) for a in (xs, ys, ds, ls))
        codes = BELT_CH[np.clip((lv * 9).astype(int), 0, len(BELT_CH) - 1)]
        # never overwrite the ring's own letters (keeps the metal crisp); particles in
        # front of the metal still show there through their glow below
        free = np.where(np.isfinite(zbuf), -np.inf, np.inf).astype(np.float32)
        self.put(x, y, codes, lv, DIAMOND, depth=d, zgrid=free)

        # glow on the brightest few (current samples only = first two blocks)
        n_now = len(self.bs_u) + len(self.belt_r)
        xn, yn, dn, ln = x[:n_now], y[:n_now], d[:n_now], lv[:n_now]
        for i in np.argsort(-ln)[:45]:
            if ln[i] > 0.35 and self.visible(xn[i], yn[i], dn[i], zbuf):
                self.glow(xn[i], yn[i], 0 if ln[i] < 0.5 else 1, 0.5 * ln[i])
        self.stats["particles"] = int((ln > 0.015).sum())

    def env(self, d, L2):
        """Studio environment seen along unit directions d (n,3)."""
        e = 0.04 + 0.22 * smoothstep(-0.1, 0.9, d[:, 1])
        for Ld, p, k in self.softboxes:
            e = e + k * np.clip(d @ Ld, 0, None) ** p
        return e + 1.2 * np.clip(d @ L2, 0, None) ** 160

    # ------------------------------------------------------------------ the ring
    def render_ring(self, t, M, cur, ghost_a, formed):
        m = self.model
        # cheap back-face cull: rotated normal's z = N . (3rd row of M); -0.3 leaves
        # margin for perspective, where the true view ray is up to ~12 deg off the z axis
        keep = np.nonzero((m.N32 @ M[2].astype(np.float32)) > -0.3)[0]
        P = m.P32[keep] @ M.T.astype(np.float32)

        sx, sy, depth = self.project(P)
        col = np.floor(sx / self.cw).astype(np.int64)
        row = np.floor(sy / self.ch).astype(np.int64)
        ok = (col >= 0) & (col < self.cols) & (row >= 0) & (row < self.rows)
        idx = row * self.cols + col
        idx, depth = idx[ok], depth[ok]
        zbuf = np.full(self.n, np.inf, np.float32)         # same dtype as depth -> fast .at
        np.minimum.at(zbuf, idx, depth)
        win = depth <= zbuf[idx]
        sel = np.nonzero(ok)[0][win]                       # winning point of each cell
        cell = idx[win]
        pk = keep[sel]

        # ---- per-winner attributes ---------------------------------------------
        N = m.N[pk] @ M.T
        Vd = norm(np.array([0.0, 0.0, self.D]) - P[sel])
        mat, ph, s, jit = m.mat[pk], m.phase[pk], m.s[pk], m.jit[pk]
        age = cur - (s + DISSOLVE * jit)                   # >= 0: built, < 0: still ghost
        solid = age >= 0

        # ---- lighting ----------------------------------------------------------
        L1 = norm([-0.45, 0.65, 0.62])                                   # key light
        L2 = norm([math.cos(t * 0.9) * 0.9, 0.35, math.sin(t * 0.9) * 0.9 + 0.25])  # moving
        nl1 = np.clip(N @ L1, 0, None)
        nl2 = np.clip(N @ L2, 0, None)
        nh1 = np.clip(np.einsum("ij,ij->i", N, norm(L1 + Vd)), 0, 1)
        nh2 = np.clip(np.einsum("ij,ij->i", N, norm(L2 + Vd)), 0, 1)
        nv = np.clip(np.einsum("ij,ij->i", N, Vd), 0, 1)
        R = 2 * nv[:, None] * N - Vd                                     # reflected view ray

        b = (0.05 + 0.30 * nl1 + 0.16 * nl2 + 0.95 * nh1 ** 40 + 0.80 * nh2 ** 70
             + 0.30 * (1 - nv) ** 3 + 0.18 * smoothstep(0.15, 0.6, R[:, 1]))
        tw = np.clip(np.sin(t * 2.6 + ph * TAU * 7), 0, None) ** 30       # facet twinkle

        dm = mat == DIAMOND
        if dm.any():
            Nd, Vv, nvd, Rd, phd = N[dm], Vd[dm], nv[dm], R[dm], ph[dm]
            fres = 0.17 + 0.83 * (1 - nvd) ** 5                          # Schlick, n = 2.42
            eta = 1 / 2.42
            kk = np.clip(1 - eta * eta * (1 - nvd * nvd), 0, None)
            T = -eta * Vv + (eta * nvd - np.sqrt(kk))[:, None] * Nd      # refracted ray
            Nm = norm(Nd * [1.0, -1.0, 1.0])                             # "opposite" facet
            T2 = T - 2 * np.einsum("ij,ij->i", T, Nm)[:, None] * Nm      # internal bounce
            scr = 0.55 * np.stack([np.sin(phd * 37.1), np.cos(phd * 53.7),
                                   np.sin(phd * 71.3 + 1.0)], 1)         # per-facet scramble
            T3 = norm(T2 + scr)
            b[dm] = (0.05 + 1.3 * fres * self.env(norm(Rd), L2) + 0.75 * self.env(T3, L2)
                     + 1.1 * nh1[dm] ** 90 + 0.9 * nh2[dm] ** 110 + 0.8 * tw[dm] * formed)

        # build-up: dissolve in, with a glowing trail behind the drawing front
        mz = smoothstep(0.0, 0.03, age)
        trail = 1.25 * np.exp(-np.clip(age, 0, None) / 0.03) * (1 - smoothstep(1.0, 1.06, cur))
        b = np.where(solid, b * mz + trail, 0.0)

        # ---- to the grid -------------------------------------------------------
        B = np.zeros(self.n, np.float32)
        MAT = np.zeros(self.n, np.uint8)
        SOL = np.zeros(self.n, bool)
        GJ = np.zeros(self.n, np.float32)
        B[cell], MAT[cell], SOL[cell], GJ[cell] = b, mat, solid, jit
        covered = np.isfinite(zbuf)
        ghost = covered & ~SOL

        bc = np.clip(B, 0, 1)
        mr = np.frombuffer(METAL_RAMP.encode(), np.uint8)
        dr = np.frombuffer(DIAM_RAMP.encode(), np.uint8)
        im = 1 + (bc ** 0.85 * (len(mr) - 2)).astype(int)
        idr = 1 + (bc * (len(dr) - 2)).astype(int)
        isd = MAT == DIAMOND
        code = np.where(isd, dr[idr], mr[im]).astype(np.uint8)
        lv = np.where(isd, 0.22 + 0.78 * bc, 0.16 + 0.84 * bc)
        pal = np.where(isd, DIAMOND, METAL).astype(np.uint8)
        pal[isd & (B > 1.0)] = GLINT

        # ghost: sparse faint dots shimmering where the ring is not built yet
        code[ghost] = GHOST_CH[(GJ[ghost] * 997).astype(int) % len(GHOST_CH)]
        lv[ghost] = ghost_a * (0.05 + 0.09 * (0.5 + 0.5 * np.sin(t * 2.5 + GJ[ghost] * 40)))
        pal[ghost] = GHOST

        # ---- outline: silhouette + depth jumps -> / \ | _ - ---------------------
        Z = zbuf.reshape(self.rows, self.cols)
        F = np.where(np.isfinite(Z), 10.0 - Z, 0.0)
        Pz = np.pad(F, 1, mode="constant")
        jump = np.maximum.reduce([np.abs(F - Pz[:-2, 1:-1]), np.abs(F - Pz[2:, 1:-1]),
                                  np.abs(F - Pz[1:-1, :-2]), np.abs(F - Pz[1:-1, 2:])])
        Pe = np.pad(F, 1, mode="edge")
        gx = (Pe[:-2, 2:] + 2 * Pe[1:-1, 2:] + Pe[2:, 2:]) - (Pe[:-2, :-2] + 2 * Pe[1:-1, :-2] + Pe[2:, :-2])
        gy = (Pe[2:, :-2] + 2 * Pe[2:, 1:-1] + Pe[2:, 2:]) - (Pe[:-2, :-2] + 2 * Pe[:-2, 1:-1] + Pe[:-2, 2:])
        gxp, gyp = (gx / self.cw).ravel(), (gy / self.ch).ravel()
        edge = SOL & (jump.ravel() > 0.35)
        ech = dir_codes(-gyp, gxp)                       # edge line is perpendicular to gradient
        ech[(ech == ord("-")) & (gyp < 0)] = ord("_")    # object above -> bottom edge
        code[edge] = ech[edge]
        lv[edge] = np.maximum(lv[edge], 0.62 * np.clip(bc[edge] * 3, 0.3, 1))

        cells = np.nonzero(covered)[0]
        self.CH[cells] = code[cells]
        self.LV[cells] = lv[cells]
        self.PAL[cells] = pal[cells]

        self.obj_lv = np.where(SOL, lv, 0).astype(np.float32)
        self.glint_cells = np.nonzero(isd & (B > 0.95) & SOL)[0]
        self.glint_b = B[self.glint_cells]
        g = np.nonzero(isd & (B > 0.8) & SOL & ~edge)[0]   # kept on top of facet lines
        self.keep_cells, self.keep_code, self.keep_lv, self.keep_pal = g, code[g], lv[g], pal[g]
        self.stats["letters"] = int(SOL.sum())

        # head of the drawing light during the setting / stone phases
        head = None
        if S_BAND <= cur < 1.0:
            near = solid & (age < 0.02)
            if near.any():
                head = (float(sx[sel][near].mean()), float(sy[sel][near].mean()))
        return zbuf, head

    def draw_facet_lines(self, M, cur, zbuf):
        """Line art of the stone's facet edges; each edge draws itself in from one end."""
        m = self.model
        frac = np.clip((cur - m.edge_s) / 0.035, 0, 1)
        on = frac > 0
        if not on.any():
            return
        A, Bv = m.edge_a[on] @ M.T, m.edge_b[on] @ M.T
        fr, es = frac[on], m.edge_s[on]
        ax, ay, _ = self.project(A)
        bx, by, _ = self.project(Bv)
        n = np.maximum(2, np.ceil(np.hypot(bx - ax, by - ay) * fr / (0.45 * self.cw)) + 1).astype(int)
        rep = np.repeat(np.arange(len(n)), n)
        k = np.arange(rep.size) - np.repeat(np.cumsum(n) - n, n)
        u = (k / (n[rep] - 1)) * fr[rep]
        P = A[rep] + (Bv - A)[rep] * u[:, None]
        x, y, d = self.project(P)
        fresh = np.exp(-np.clip(cur - es, 0, None) / 0.05)
        lv = (0.36 + 0.6 * fresh)[rep]
        self.put(x, y, dir_codes(bx - ax, by - ay)[rep], lv, DIAMOND,
                 depth=d, zgrid=zbuf + 0.045)
        # glowing tips of edges that are still being drawn
        tip = fr < 1
        for tx, ty in zip((ax + (bx - ax) * fr)[tip], (ay + (by - ay) * fr)[tip]):
            self.glow(tx, ty, 1, 0.5)
        # glints stay on top of the line art
        c = self.keep_cells
        self.CH[c], self.LV[c], self.PAL[c] = self.keep_code, self.keep_lv, self.keep_pal

    # ------------------------------------------------------------------ one frame
    def update(self, t, dt):
        self.reset_layers()
        self.glows = []
        x = (t - FORM_START) / FORM_DUR
        cur = x if (x < 0 or x >= 1) else ease(x)          # drawing progress
        ghost_a = 0.9 * float(smoothstep(0.0, GHOST_IN, t))
        rot = max(0.0, t - ROT_START) * ROT_SPEED
        tilt = math.radians(6.0) * math.sin(t * 0.35)
        M = rot_x(tilt) @ rot_y(math.radians(rot))
        self.stats["rotation"] = rot % 360.0
        formed = float(smoothstep(FORM_END - 0.3, FORM_END + 1.5, t))

        # 1) dust
        dy = (self.dust_y - t * self.dust_v) % self.H
        tw = np.clip(np.sin(t * self.dust_w + self.dust_ph), 0, None) ** 3
        self.put(self.dust_x, dy, self.dust_ch,
                 (0.05 + 0.32 * tw) * smoothstep(0.0, 2.0, t), self.dust_pal)

        # 2) orbit paths (faint dashed ellipses)
        oa = float(smoothstep(0.5, 2.5, t))
        th = np.linspace(0, TAU, 260, endpoint=False)[::2]
        for e1, e2, rad, _, _ in self.orbits:
            P = rad * (np.cos(th)[:, None] * e1 + np.sin(th)[:, None] * e2)
            px, py, _ = self.project(P)
            self.put(px, py, dir_codes(np.roll(px, -1) - px, np.roll(py, -1) - py),
                     0.11 * oa, ORBIT)

        # 3) ring + facet line art
        zbuf, head = self.render_ring(t, M, cur, ghost_a, formed)
        self.draw_facet_lines(M, cur, zbuf)

        dc = self.model.diamond_c @ M.T
        ox, oy, _ = self.project(dc[None])
        ox, oy = float(ox[0]), float(oy[0])

        # 4) light rays from the stone (only into empty cells -> behind the ring)
        if formed > 0:
            for a, L, k, sp, phs in zip(self.ray_ang, self.ray_len, self.ray_int,
                                        self.ray_spd, self.ray_ph):
                Lk = L * (0.7 + 0.3 * math.sin(t * sp + phs))
                d = np.arange(self.cw * 4.0, Lk, self.cw * 0.55)
                dx, dy_ = math.cos(a), -math.sin(a)
                lv = formed * k * (1 - d / Lk) ** 1.4 * (0.75 + 0.25 * math.sin(t * 7 + phs))
                self.put(ox + d * dx, oy + d * dy_, dir_codes(dx, dy_), lv, RAY, under=True)
            for sgn in (-1, 1):                          # dashed horizontal flare
                d = np.arange(self.cw * 6, self.cw * 70, self.cw * 2.0)
                lv = formed * 0.22 * (1 - d / (self.cw * 70)) * (0.7 + 0.3 * math.sin(t * 1.3))
                self.put(ox + sgn * d, np.full_like(d, oy), ord("-"), lv, RAY, under=True)

        # 5) completion burst: an expanding ring of light when the stone is finished
        tau = t - FORM_END
        if 0 <= tau < 1.6:
            q = tau / 1.6
            rad = 20 + 0.55 * self.W * (1 - (1 - q) ** 3)
            n = int(TAU * rad / (self.cw * 1.1))
            a = np.linspace(0, TAU, n, endpoint=False)
            codes = np.where(np.arange(n) % 3 == 0, ord("*"), ord(".")).astype(np.uint8)
            self.put(ox + rad * np.cos(a), oy + 0.85 * rad * np.sin(a), codes,
                     0.6 * (1 - q) ** 1.5, RAY, under=True)
            self.glow(ox, oy, 7, 0.75 * (1 - q))
            if tau < 1.0:
                self.glow(ox, oy, 4, 1.0 - tau, kind=1)

        # 6) floor reflection: mirror the ring's rows below the floor line, dimmed
        bot = self.project(np.array([[0.0, self.model.bottom_y - 0.25, 0.0]]) @ M.T)[1][0]
        fr = int(bot / self.ch)
        O = self.obj_lv.reshape(self.rows, self.cols)
        rr = np.frombuffer(REFL_RAMP.encode(), np.uint8)
        depth_rows = 11
        for k in range(1, depth_rows):
            r_dst, r_src = fr + k, fr - k - 3
            if r_dst >= self.rows or r_src < 0:
                break
            shift = int(round(math.sin(r_dst * 0.9 + t * 2.0) * 0.7))
            lvr = np.roll(O[r_src], shift) * 0.30 * (1 - k / depth_rows) ** 1.5
            cols = np.nonzero(lvr > 0.02)[0]
            if len(cols):
                ci = np.clip((lvr[cols] / 0.30 * (len(rr) - 1)).astype(int), 1, len(rr) - 1)
                self.put(cols * self.cw + 1, np.full(len(cols), r_dst * self.ch + 1),
                         rr[ci], lvr[cols], REFLECT, under=True)

        # 7) orbit particles with trails (z-tested against the ring)
        kk = np.arange(len(TRAIL_CH))
        for e1, e2, rad, spd, npart in self.orbits:
            for j in range(npart):
                thk = t * spd + j * TAU / npart - np.sign(spd) * kk * 0.045
                P = rad * (np.cos(thk)[:, None] * e1 + np.sin(thk)[:, None] * e2)
                px, py, dpt = self.project(P)
                lv = oa * 0.75 * (1 - kk / len(TRAIL_CH)) ** 1.2 * (0.6 + 0.4 * (P[:, 2] > 0))
                self.put(px, py, TRAIL_CH, lv, ORBIT, depth=dpt, zgrid=zbuf)
                if self.visible(px[0], py[0], dpt[0], zbuf):
                    self.glow(px[0], py[0], 2, 0.55 * oa)

        # 8) glowing motes (bokeh: near = bigger, softer)
        ma = float(smoothstep(0.3, 3.0, t))
        ang = self.mote_a + t * self.mote_w
        P = np.stack([self.mote_r * np.cos(ang),
                      self.mote_y + self.mote_bob * np.sin(t * 0.7 + self.mote_ph),
                      self.mote_r * np.sin(ang)], 1)
        px, py, dpt = self.project(P)
        twm = 0.45 + 0.55 * np.clip(np.sin(t * self.mote_tw + self.mote_ph), 0, None) ** 2
        near = np.clip((self.D - dpt + 3.2) / 6.4, 0, 1)
        km = ma * twm * (0.18 + 0.30 * near)
        self.put(px, py, ord("."), km * 0.9, DIAMOND, depth=dpt, zgrid=zbuf)
        for i in range(len(px)):
            if self.visible(px[i], py[i], dpt[i], zbuf):
                self.glow(px[i], py[i], int(1 + near[i] * 3), km[i] * (0.8 - 0.4 * near[i]))

        # 8b) particles flowing in ring-shaped regions (band stream + orbital belt)
        self.particle_rings(t, M, cur, zbuf)

        # 9) the drawing lights: two on the band, then one rising through the setting
        heads = []
        if 0 <= cur < S_BAND:
            th = cur / S_BAND * math.pi
            for u in (-math.pi / 2 + th, -math.pi / 2 - th):
                p = np.array([[math.cos(u), math.sin(u) - self.model.cy, 0.0]]) @ M.T
                hx, hy, _ = self.project(p)
                heads.append((float(hx[0]), float(hy[0])))
        elif head is not None:
            heads.append(head)
        for hx, hy in heads:
            self.star(hx, hy, 1.0)
            self.glow(hx, hy, 4, 0.55)
            self.glow(hx, hy, 2, 0.8, kind=1)
            self.sparks.emit(hx, hy, self.rng.poisson(110 * dt))

        if formed > 0.5 and len(self.glint_cells):
            n = self.rng.poisson(30 * dt)
            if n:
                c = self.rng.choice(self.glint_cells, n)
                self.sparks.emit((c % self.cols) * self.cw, (c // self.cols) * self.ch, n,
                                 20, 120, 40)
        self.sparks.step(dt)
        if len(self.sparks.x):
            f = self.sparks.life / self.sparks.max
            self.put(self.sparks.x, self.sparks.y, SPARK_CH[np.clip((f * 4).astype(int), 0, 3)],
                     0.25 + 0.75 * f, SPARK)
            for sx, sy, ff in zip(self.sparks.x.tolist(), self.sparks.y.tolist(), f.tolist()):
                if ff > 0.25:
                    self.glow(sx, sy, 0, 0.12 + 0.3 * ff)

        # 10) diamond glints -> star flares on the brightest few cells
        if len(self.glint_cells) and cur > 0.9:
            for c, b in zip(self.glint_cells[np.argsort(-self.glint_b)[:6]],
                            self.glint_b[np.argsort(-self.glint_b)[:6]]):
                gx, gy = (c % self.cols + 0.5) * self.cw, (c // self.cols + 0.5) * self.ch
                kk2 = float(np.clip(b - 0.9, 0, 0.8) / 0.8)
                self.glow(gx, gy, 1 + int(kk2 * 2), 0.18 + 0.3 * kk2)
                if kk2 > 0.3:
                    self.glow(gx, gy, int(kk2 * 4.4), 0.25 + 0.45 * kk2, kind=1)
        if formed > 0:
            self.glow(ox, oy, 7, 0.22 * formed)
        self.stats["glints"] = int(len(self.glint_cells) + len(self.sparks.x))

    # ------------------------------------------------------------------ drawing
    def glyph(self, key):
        g = self.glyphs.get(key)
        if g is None:
            code, pal, lvl = key >> 12, (key >> 4) & 0xFF, key & 0xF
            k = lvl / 15.0
            col = tuple(int(c * (0.04 + 0.96 * k ** 1.25)) for c in PALETTES[pal])
            g = self.font.render(chr(code), True, col, BG).convert()
            self.glyphs[key] = g
        return g

    def draw(self, screen, t):
        screen.fill(BG)
        nz = np.nonzero((self.CH != 32) & (self.LV > 0.02))[0]
        L = np.clip((self.LV[nz] * 15 + 0.5).astype(np.int32), 1, 15)
        keys = (self.CH[nz].astype(np.int32) << 12) | (self.PAL[nz].astype(np.int32) << 4) | L
        get, pos, make = self.glyphs.get, self.pos, self.glyph
        seq = []
        for k, i in zip(keys.tolist(), nz.tolist()):
            g = get(k)
            seq.append((g if g is not None else make(k), pos[i]))
        screen.blits(seq, doreturn=False)

        add = pygame.BLEND_RGB_ADD                       # glowing particles
        for x, y, kind, size, lvl in self.glows:
            spr = self.glow_sprite(kind, size, lvl)
            w, h = spr.get_size()
            screen.blit(spr, (int(x - w // 2), int(y - h // 2)), special_flags=add)

        if self.bloom:
            # blur by rescaling, mixed at 1/5 size. The glow is soft and changes slowly,
            # so it is rebuilt every 2nd frame and reused in between (halves its cost).
            self.frame_no = getattr(self, "frame_no", 0) + 1
            if self.frame_no % 2 or getattr(self, "bloom_surf", None) is None:
                W, H = self.W, self.H
                sw, sh = W // 5, H // 5
                small = pygame.transform.smoothscale(screen, (sw, sh))
                tiny = pygame.transform.smoothscale(small, (W // 20, H // 20))
                wide = pygame.transform.smoothscale(tiny, (sw, sh))
                small.fill((150, 150, 150), special_flags=pygame.BLEND_RGB_MULT)
                for _ in range(2):
                    small.blit(wide, (0, 0), special_flags=add)
                self.bloom_surf = pygame.transform.smoothscale(small, (W, H))
            screen.blit(self.bloom_surf, (0, 0), special_flags=add)

        if self.hud:
            st = self.stats
            lines = ["ascii diamond ring",
                     f"rotation {st['rotation']:06.2f}°   glints {st['glints']:>4}",
                     f"letters {st['letters']:,}   particles {st['particles']:,}"]
            lh = int(self.hud_font.get_linesize() * 1.6)
            y = self.H - lh * len(lines) - 14
            for s in lines:
                screen.blit(self.hud_font.render(s, True, HUD_GREY), (60, y))
                y += lh
            txt = self.hud_font.render(f"t {t:05.2f}s", True, HUD_HI)
            x = self.W - txt.get_width() - 60
            screen.blit(txt, (x, 30))
            if int(t * 2) % 2 == 0:
                pygame.draw.circle(screen, HUD_HI, (x - 12, 30 + txt.get_height() // 2), 4)


# ============================================================================ main
def main():
    ap = argparse.ArgumentParser(description="ASCII diamond ring animation")
    ap.add_argument("--windowed", action="store_true", help="window instead of fullscreen")
    ap.add_argument("--width", type=int, default=1080, help="window / offline width")
    ap.add_argument("--height", type=int, default=960, help="window / offline height")
    ap.add_argument("--font", type=int, default=0, help="glyph size px (0 = auto from height)")
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--no-bloom", action="store_true")
    ap.add_argument("--headless", action="store_true", help="no window (for --snap/--record)")
    ap.add_argument("--snap", default="", help="comma list of times (s) to save as PNG, then exit")
    ap.add_argument("--record", type=float, default=0.0, help="save every frame for N seconds")
    ap.add_argument("--out", default="frames")
    args = ap.parse_args()

    snaps = sorted(float(x) for x in args.snap.split(",") if x.strip())
    offline = bool(snaps or args.record)
    if args.headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    elif sys.platform == "win32":                    # real pixels, not DPI-stretched ones
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            pass

    pygame.init()
    if args.headless or args.windowed or offline:
        screen = pygame.display.set_mode((args.width, args.height))
    else:
        info = pygame.display.Info()
        screen = pygame.display.set_mode((info.current_w, info.current_h), pygame.FULLSCREEN)
        pygame.mouse.set_visible(False)
    pygame.display.set_caption("ASCII Diamond Ring  -  ESC quit | SPACE pause | R restart")
    W, H = screen.get_size()
    font = args.font or max(8, round(H / 105))
    scene = Scene(W, H, font, bloom=not args.no_bloom)
    fps = 30 if offline else args.fps

    if offline:
        os.makedirs(args.out, exist_ok=True)
    clock = pygame.time.Clock()
    t, frame, paused = 0.0, 0, False

    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                pygame.quit()
                return
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_SPACE:
                    paused = not paused
                elif e.key == pygame.K_r:
                    t = 0.0
                    scene.sparks.clear()
                elif e.key == pygame.K_b:
                    scene.bloom = not scene.bloom
                elif e.key == pygame.K_h:
                    scene.hud = not scene.hud
                elif e.key == pygame.K_s:
                    name = f"ring_{int(t * 100):05d}.png"
                    pygame.image.save(screen, name)
                    print("saved", name)

        dt = 1.0 / fps if offline else min(clock.tick(fps) / 1000.0, 0.05)
        if not paused:
            t += dt
            scene.update(t, dt)
        scene.draw(screen, t)
        pygame.display.flip()

        if offline:
            if args.record:
                pygame.image.save(screen, os.path.join(args.out, f"f{frame:05d}.png"))
                if t >= args.record:
                    break
            while snaps and t >= snaps[0]:
                name = os.path.join(args.out, f"snap_{snaps.pop(0):06.2f}s.png")
                pygame.image.save(screen, name)
                print("saved", name)
            if not snaps and not args.record:
                break
        frame += 1
    pygame.quit()


if __name__ == "__main__":
    sys.exit(main())
