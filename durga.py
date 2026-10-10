"""
Maa Durga - Live Sketch Studio   (no image file needed!)
Three artworks are built into this file. Just run:

    python durga.py

In the app: pick an artwork from the dropdown (or keys 1 / 2 / 3), or click
"Open image" to sketch any picture of your own.

Modes
  sketch : grey pencil outlines -> colour fills -> dark ink   (colour art)
  pencil : tonal shading + ink line-work                      (grey art)
  color  : direct colour layers, light -> dark

Optional:  python durga.py my.png --mode sketch --colors 7 --width 700
           python durga.py --export-images   (saves durga.jpg, face.png, lion.png next to this file)
Shortcuts: C change colour theme | Space play/pause | R restart | S save PNG | End skip | Up/Down speed | Esc quit
"""
import argparse
import base64
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import turtle

import cv2
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))

# ---- UI theme ---------------------------------------------------------
BAR_BG = "#1c1410"
BAR_FG = "#f5e6d3"
ACCENT = "#E8590C"
ACCENT_HOVER = "#ff7a2e"
FONT = ("Segoe UI", 10)
PENCIL = "#6e6e6e"

# built-in artworks: key -> (title, mode, colour layers)
ARTS = {
    "face": ("1. Durga Face - pencil + colour", "sketch", 7, "Original"),
    "lion": ("2. Durga on Lion - pencil drawing", "pencil", 5, "Saffron & Gold"),
    "silhouette": ("3. Durga & Mahishasura - silhouette", "color", 6, "Original"),
}

# colour themes: dark -> light gradient stops (None = keep the artwork's own colours)
THEMES = {
    "Original": None,
    "Saffron & Gold": ["#2b1408", "#8a3b0f", "#e8590c", "#f6a21e", "#ffe9b0"],
    "Navratri Red": ["#2a0a0f", "#7a1020", "#c8202f", "#f2735a", "#ffe6d5"],
    "Royal Blue & Gold": ["#0b1330", "#1f3a8a", "#3f6fd1", "#e8b64a", "#fff3d1"],
    "Peacock Teal": ["#06222a", "#0b5d6b", "#13a3a1", "#f4c95d", "#fff6e0"],
}


# ---- Image helpers ----------------------------------------------------
def load_image(path):
    for p in (path, os.path.join(BASE, path)):
        if os.path.isfile(p):
            data = np.fromfile(p, dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img is not None:
                return img
    return None


def embedded_image(key):
    raw = base64.b64decode("".join(ART_DATA[key].split()))
    return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)


def export_images(folder=BASE):
    """Save the built-in artworks as normal image files (so older scripts can use them)."""
    names = {"silhouette": "durga.jpg", "face": "face.png", "lion": "lion.png"}
    for key, name in names.items():
        img = embedded_image(key)
        ext = os.path.splitext(name)[1]
        params = [cv2.IMWRITE_JPEG_QUALITY, 95] if ext == ".jpg" else []
        ok, buf = cv2.imencode(ext, img, params)
        if ok:
            buf.tofile(os.path.join(folder, name))
            print("saved", os.path.join(folder, name))


def hex_rgb(bgr):
    b, g, r = [int(v) for v in bgr]
    return f"#{r:02x}{g:02x}{b:02x}"


def hex_to_bgr(h):
    return np.array([int(h[5:7], 16), int(h[3:5], 16), int(h[1:3], 16)], np.float32)


def apply_theme(centers, lum, stops):
    """Re-colour by brightness: darkest tone -> first stop, lightest -> last stop."""
    cols = np.array([hex_to_bgr(h) for h in stops])
    t = (lum - lum.min()) / max(1e-6, float(lum.max() - lum.min()))
    pos = t * (len(cols) - 1)
    lo = np.clip(np.floor(pos).astype(int), 0, len(cols) - 2)
    f = (pos - lo)[:, None]
    return (cols[lo] * (1 - f) + cols[lo + 1] * f).astype(np.float32), cols[0]


def is_pencil(img):
    """True for (nearly) grayscale artwork such as pencil drawings."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return float(hsv[..., 1].mean()) < 25


def mask_to_shapes(mask, col, eps=0.6, min_area=25):
    """Binary mask -> fill shapes (outer contour + holes)."""
    cnts, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hier is None:
        return []
    hier = hier[0]
    group = []
    for i, c in enumerate(cnts):
        if hier[i][3] != -1 or cv2.contourArea(c) < min_area:
            continue
        outer = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
        if len(outer) < 3:
            continue
        holes = []
        j = hier[i][2]
        while j != -1:
            if cv2.contourArea(cnts[j]) >= 8:
                hp = cv2.approxPolyDP(cnts[j], eps, True).reshape(-1, 2)
                if len(hp) >= 3:
                    holes.append(hp)
            j = hier[j][0]
        group.append(dict(kind="fill", color_hex=hex_rgb(col), color_bgr=col,
                          outer=outer, holes=holes, area=cv2.contourArea(c)))
    group.sort(key=lambda s: -s["area"])
    return group


def pencil_strokes(fills, rng, min_area=60):
    """Wobbly grey outlines of every fill shape = the 'pencil sketch' pass."""
    strokes = []
    for s in sorted(fills, key=lambda s: -s["area"]):
        if s["area"] < min_area:
            continue
        for poly in [s["outer"]] + list(s["holes"]):
            if len(poly) < 4:
                continue
            pts = poly.astype(np.float32) + rng.normal(0, 0.7, poly.shape)
            strokes.append(dict(kind="stroke", color_hex=PENCIL, width=1.2,
                                pts=np.vstack([pts, pts[:1]])))
    return strokes


# ---- Image -> drawable shapes ----------------------------------------
def build_layers(img, width, k, mode="auto", theme=None):
    """Return (w, h, bg_bgr, shapes). Shapes are drawn in list order."""
    h0, w0 = img.shape[:2]
    w, h = width, int(width * h0 / w0)
    interp = cv2.INTER_CUBIC if w > w0 else cv2.INTER_AREA
    resized = cv2.resize(img, (w, h), interpolation=interp)

    small = cv2.resize(resized, (160, max(1, int(160 * h / w))))
    cv2.setRNGSeed(7)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, _, centers = cv2.kmeans(small.reshape(-1, 3).astype(np.float32), k,
                               None, crit, 3, cv2.KMEANS_PP_CENTERS)
    dist = ((resized.reshape(-1, 1, 3).astype(np.float32) - centers[None]) ** 2).sum(2)
    raw = dist.argmin(1).reshape(h, w)

    sigma = max(1.2, w / 400)
    scores = np.stack([cv2.GaussianBlur((raw == i).astype(np.float32), (0, 0), sigma)
                       for i in range(k)], axis=-1)
    labels = scores.argmax(-1).astype(np.uint8)

    lum = centers @ np.array([0.114, 0.587, 0.299])
    ink_col = np.array([32, 32, 32], np.float32)
    if theme:
        centers, ink_col = apply_theme(centers, lum, theme)
    order = np.argsort(-lum)            # light -> dark : background first, ink last
    bg = centers[order[0]]

    kernel = np.ones((3, 3), np.uint8)
    fills = []
    for idx in order[1:]:
        mask = ((labels == idx) * 255).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.dilate(mask, kernel)      # 1px overlap hides seams between colours
        fills.extend(mask_to_shapes(mask, centers[idx]))

    if mode == "auto":
        mode = "pencil" if is_pencil(img) else "sketch"

    if mode == "pencil":
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (0, 0), 0.8)
        ink = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 15, 9)
        ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        fills.extend(mask_to_shapes(ink, ink_col,
                                    eps=0.5, min_area=6))

    if mode == "sketch":
        strokes = pencil_strokes(fills, np.random.default_rng(3))
        return w, h, bg, strokes + fills      # pencil first, then colour + ink
    return w, h, bg, fills


# ---- App ---------------------------------------------------------------
class SketchApp:
    def __init__(self, args):
        self.args = args
        self.screen = turtle.Screen()
        self.canvas = self.screen.getcanvas()
        self.root = self.canvas.winfo_toplevel()
        self.screen.tracer(0)

        self.pen = turtle.Turtle()
        self.pen.speed(0)
        self.pen.shape("circle")
        self.pen.shapesize(0.35)
        self.pen.penup()

        self.speed = tk.IntVar(value=14)
        self.status = tk.StringVar()
        self.pct = tk.DoubleVar(value=0)
        self.art_var = tk.StringVar()
        self.theme_var = tk.StringVar(value="Original")
        self.cur = None
        self.running = False
        self.after_id = None
        self.done_pts = 0
        self.gen = iter(())
        self.shapes, self.n_strokes, self.total = [], 0, 1
        self.w = self.h = 0
        self.bg = np.array([255, 255, 255], np.float32)

        self.build_toolbar()
        self.bind_keys()
        self.root.protocol("WM_DELETE_WINDOW", self.quit)

        img, mode, colors, title = self.initial_art()
        self.set_art(img, mode, colors, title)
        self.restart()

    # -- choosing artwork
    def initial_art(self):
        a = self.args
        if a.image:
            img = load_image(a.image)
            if img is not None:
                self.theme_var.set(a.theme or "Original")
                return img, a.mode or "auto", a.colors or 6, os.path.basename(a.image)
            messagebox.showwarning("Image not found",
                                   f"Could not open '{a.image}'.\nShowing built-in artwork instead.")
        title, mode, colors, theme = ARTS["face"]
        self.theme_var.set(a.theme or theme)
        return embedded_image("face"), a.mode or mode, a.colors or colors, title

    def pick_preset(self, _e=None):
        key = list(ARTS)[self.combo.current()]
        title, mode, colors, theme = ARTS[key]
        self.theme_var.set(self.args.theme or theme)
        self.set_art(embedded_image(key), self.args.mode or mode,
                     self.args.colors or colors, title)
        self.restart()
        self.root.focus_set()

    def pick_number(self, i):
        self.combo.current(i)
        self.pick_preset()

    def open_image(self, _e=None):
        was_running, self.running = self.running, False
        path = filedialog.askopenfilename(
            title="Choose an image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.webp"), ("All files", "*.*")])
        img = load_image(path) if path else None
        if img is None:
            self.running = was_running
            if path:
                messagebox.showerror("Error", "That file could not be read as an image.")
            return
        self.theme_var.set(self.args.theme or "Original")
        self.set_art(img, self.args.mode or "auto", self.args.colors or 6,
                     os.path.basename(path))
        self.restart()

    def pick_theme(self, _e=None):
        if self.cur:
            self.set_art(*self.cur)
            self.restart()
        self.root.focus_set()

    def cycle_theme(self, _e=None):
        names = list(THEMES)
        self.theme_var.set(names[(names.index(self.theme_var.get()) + 1) % len(names)])
        self.pick_theme()

    def set_art(self, img, mode, colors, title):
        self.cur = (img, mode, colors, title)
        ratio = img.shape[0] / img.shape[1]
        avail_h = self.root.winfo_screenheight() - 230
        width = min(self.args.width, int(avail_h / ratio))
        width = max(width, 300)
        self.w, self.h, self.bg, self.shapes = build_layers(img, width, colors, mode,
                                                           THEMES.get(self.theme_var.get()))
        if not self.shapes:
            messagebox.showerror("Nothing to draw", "No shapes were detected in this image.")
            return
        self.total = max(1, sum(self.npts(s) for s in self.shapes))
        self.n_strokes = sum(1 for s in self.shapes if s["kind"] == "stroke")
        self.screen.setup(max(self.w + 40, 560), self.h + 175)
        self.screen.title(f"Maa Durga - Live Sketch Studio  |  {title}")
        self.screen.bgcolor(hex_rgb(self.bg))
        self.art_var.set(title)

    @staticmethod
    def npts(s):
        if s["kind"] == "stroke":
            return len(s["pts"])
        return len(s["outer"]) + sum(len(x) for x in s["holes"])

    def tt(self, p):
        return (p[0] - self.w / 2, self.h / 2 - p[1])

    # -- drawing generator (one yield per point -> smooth pause/speed)
    def draw_gen(self):
        pen = self.pen
        n_fill = len(self.shapes) - self.n_strokes
        for si, s in enumerate(self.shapes):
            if s["kind"] == "stroke":
                self.status.set(f"Pencil sketch {si + 1}/{self.n_strokes}")
                pen.pensize(s["width"])
                pen.color(s["color_hex"])
                pts = [self.tt(p) for p in s["pts"]]
                pen.penup(); pen.goto(pts[0]); pen.pendown()
                for p in pts[1:]:
                    pen.goto(p)
                    self.done_pts += 1
                    yield
                pen.penup()
                continue

            self.status.set(f"Colouring {si - self.n_strokes + 1}/{n_fill}")
            pen.pensize(1)
            pen.color(s["color_hex"], s["color_hex"])
            o = [self.tt(p) for p in s["outer"]]
            pen.penup(); pen.goto(o[0]); pen.pendown()
            pen.begin_fill()
            for p in o[1:]:
                pen.goto(p)
                self.done_pts += 1
                yield
            pen.goto(o[0])
            for hole in s["holes"]:
                hp = [self.tt(p) for p in hole]
                pen.penup()
                pen.goto(hp[0])           # bridge in: pen up so no visible line;
                pen.pendown()             # it cancels out in the even-odd fill
                for p in hp[1:]:
                    pen.goto(p)
                    self.done_pts += 1
                    yield
                pen.goto(hp[0])
                pen.penup()
                pen.goto(o[0])            # bridge out
                pen.pendown()
            pen.end_fill()
            pen.penup()

    # -- main loop
    def tick(self):
        if self.running:
            try:
                for _ in range(self.speed.get()):
                    next(self.gen)
            except StopIteration:
                self.finish()
                return
            self.refresh()
        self.after_id = self.root.after(16, self.tick)

    def refresh(self):
        self.pct.set(100 * self.done_pts / self.total)
        self.screen.update()

    def finish(self):
        self.running = False
        self.pen.hideturtle()
        self.pct.set(100)
        self.status.set("Done - press S to save, R to replay")
        self.btn_play.config(text="Play")
        self.screen.update()

    # -- actions
    def toggle(self, _e=None):
        if not self.pen.isvisible():      # finished -> replay
            return self.restart()
        self.running = not self.running
        self.btn_play.config(text="Pause" if self.running else "Play")
        if not self.running:
            self.status.set("Paused")

    def restart(self, _e=None):
        if self.after_id:
            self.root.after_cancel(self.after_id)
        self.pen.end_fill()
        self.pen.clear()
        self.pen.penup()
        self.pen.showturtle()
        self.done_pts = 0
        self.gen = self.draw_gen()
        self.running = True
        self.btn_play.config(text="Pause")
        self.tick()

    def skip(self, _e=None):
        if self.after_id:
            self.root.after_cancel(self.after_id)
        for _ in self.gen:
            pass
        self.finish()

    def save(self, _e=None):
        scale = 2
        out = np.empty((self.h * scale, self.w * scale, 3), np.uint8)
        out[:] = self.bg.astype(np.uint8)
        for s in self.shapes:
            if s["kind"] != "fill":
                continue
            polys = [np.round(s["outer"] * scale).astype(np.int32)]
            polys += [np.round(hh * scale).astype(np.int32) for hh in s["holes"]]
            cv2.fillPoly(out, polys, [int(v) for v in s["color_bgr"]], cv2.LINE_AA)
        path = filedialog.asksaveasfilename(defaultextension=".png",
                                            initialfile="durga_art.png",
                                            filetypes=[("PNG image", "*.png")])
        if not path:
            return
        ok, buf = cv2.imencode(".png", out)
        if ok:
            buf.tofile(path)
            self.status.set(f"Saved: {os.path.basename(path)}")

    def change_speed(self, delta):
        self.speed.set(max(1, min(80, self.speed.get() + delta)))

    def quit(self, _e=None):
        if self.after_id:
            self.root.after_cancel(self.after_id)
        self.root.destroy()

    # -- UI
    def build_toolbar(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("D.Horizontal.TProgressbar", troughcolor="#3a2a20",
                        background=ACCENT, bordercolor=BAR_BG,
                        lightcolor=ACCENT, darkcolor=ACCENT)

        bar = tk.Frame(self.root, bg=BAR_BG, pady=6)
        try:
            bar.pack(side="bottom", fill="x", before=self.canvas.master)
        except tk.TclError:
            bar.pack(side="bottom", fill="x")
        row1 = tk.Frame(bar, bg=BAR_BG)
        row1.pack(fill="x", padx=6)
        row2 = tk.Frame(bar, bg=BAR_BG)
        row2.pack(fill="x", padx=10, pady=(6, 0))

        def button(text, cmd):
            b = tk.Button(row1, text=text, command=cmd, bg=ACCENT, fg="white",
                          activebackground=ACCENT_HOVER, activeforeground="white",
                          relief="flat", bd=0, padx=10, pady=4, font=FONT,
                          cursor="hand2")
            b.pack(side="left", padx=3)
            return b

        self.btn_play = button("Pause", self.toggle)
        button("Restart", self.restart)
        button("Skip", self.skip)
        button("Save PNG", self.save)
        button("Open image", self.open_image)

        tk.Label(row1, text="Speed", bg=BAR_BG, fg=BAR_FG, font=FONT).pack(side="left", padx=(12, 2))
        tk.Scale(row1, from_=1, to=80, orient="horizontal", variable=self.speed,
                 length=90, showvalue=False, bg=BAR_BG, fg=BAR_FG,
                 troughcolor="#3a2a20", highlightthickness=0,
                 activebackground=ACCENT).pack(side="left")

        self.combo = ttk.Combobox(row2, textvariable=self.art_var, state="readonly",
                                  values=[v[0] for v in ARTS.values()], width=30)
        self.combo.pack(side="left")
        self.combo.bind("<<ComboboxSelected>>", self.pick_preset)
        tk.Label(row2, text="Colour", bg=BAR_BG, fg=BAR_FG, font=FONT).pack(side="left", padx=(8, 2))
        self.theme_combo = ttk.Combobox(row2, textvariable=self.theme_var, state="readonly",
                                        values=list(THEMES), width=16)
        self.theme_combo.pack(side="left")
        self.theme_combo.bind("<<ComboboxSelected>>", self.pick_theme)

        row3 = tk.Frame(bar, bg=BAR_BG)
        row3.pack(fill="x", padx=10, pady=(6, 0))
        right = tk.Frame(row3, bg=BAR_BG)
        right.pack(fill="x", expand=True)
        tk.Label(right, textvariable=self.status, bg=BAR_BG, fg=BAR_FG,
                 font=FONT, anchor="w").pack(side="top", fill="x")
        ttk.Progressbar(right, variable=self.pct, maximum=100,
                        style="D.Horizontal.TProgressbar").pack(side="top", fill="x", pady=(2, 0))

    def bind_keys(self):
        r = self.root
        r.bind("<space>", self.toggle)
        r.bind("<r>", self.restart)
        r.bind("<s>", self.save)
        r.bind("<o>", self.open_image)
        r.bind("<c>", self.cycle_theme)
        r.bind("<End>", self.skip)
        r.bind("<Up>", lambda e: self.change_speed(4))
        r.bind("<Down>", lambda e: self.change_speed(-4))
        r.bind("<Escape>", self.quit)
        for i in range(len(ARTS)):
            r.bind(str(i + 1), lambda e, i=i: self.pick_number(i))


def main():
    ap = argparse.ArgumentParser(description="Maa Durga live sketch studio")
    ap.add_argument("image", nargs="?", default=None,
                    help="optional image path (built-in artwork is used if omitted)")
    ap.add_argument("--width", type=int, default=720, help="canvas width in px")
    ap.add_argument("--colors", type=int, default=None, help="colour layers (3-9)")
    ap.add_argument("--mode", choices=["auto", "sketch", "pencil", "color"], default=None,
                    help="sketch = pencil then colour; pencil = shading + ink; color = layers")
    ap.add_argument("--theme", choices=list(THEMES), default=None,
                    help="colour theme (default depends on the artwork)")
    ap.add_argument("--export-images", action="store_true",
                    help="save durga.jpg, face.png, lion.png next to this file and exit")
    args = ap.parse_args()
    if args.export_images:
        export_images()
        return
    if args.colors:
        args.colors = max(3, min(9, args.colors))
    SketchApp(args)
    turtle.mainloop()


# ---- Built-in artwork (base64 images) ---------------------------------
ART_DATA = {
    "face": """
iVBORw0KGgoAAAANSUhEUgAAASYAAAFeCAMAAAAfa0JEAAAASFBMVEX///////j7//3+/Pz2+PfnwYX3pCjnpT32oCvwli/dkz+7
jFOdVTW1KSxVOSArFw4yBgQcBQQTCggTAQIHCgkGBAUFAQgHAABNBOOTAABtSUlEQVR42u1dC3vbtrKUJdKUCBQgwdf//6d3Z2ZB
UrLTuj1Nb5JW53xtmsS2tFws9jE7c7n8/77e7HW7XPG6XG635vbW5NzYv2/N21vTvF3+ex1mgp2aa9OYfZpUtmj/ulzNTm+X5j8T
uZ1u+NcVVmnemlvM27a2l+Z6fYMv/WcmvW43/ctMBDu1W1nnNTc4cDiG/5npbKs3BCKzU1vmdZ7nJTU4gf9Z6cWjGK7NMnmZV/Om
xexk/3mBqf57KSgpjlsgsvA9z2uhOy3BQrpF8v/cqZrpeuNtZzlAMkcqZdvyvBSzE8LWfxmBXg0SAXMmS5QifKmMyVKCsqxrMNPZ
0VOA+rcfPhoJZmqiRaW5TKkNeZzKmtfwRgMiRP3nTRdkAvZvs1KBL4XQxsHsVPIamYfj7N3+7d50ozPBl0qexmkIXd93cZhGc6c5
4NDBl5r/zMTsO1lOWaZxiB1fZqdsd94WmBSYof7tZmqYWkbccSOs9Lg/3rsO566s64o88z9fYmoJX7Ksci5jNl8yKz0eXd/Cn/K6
zfCnf2tsQk55HDpklfOymS+13fv74w47uT/NZUuMTteLei3/Hu9Re6nhFXdjXNrKvJUtm5XsyD0ed3s9zJ+mkrd5m5IFJ6ZX/LJ/
TZHrZrowJ2qatizzto0TTpwdua7vOpjKHCpmM55l5bk1I12b5vrvcSjc/vy43l+6tuZJS8mI3pYJmJFCMnv1bqepbHNZS8P2AVp1
/5ZzV+3E1sn12lplki0updDTSn1Iw5CC+RXtZGnmkmfaCVa9Nf8md9KLJy5v2Wpdyyp7O2l918c8btOUQ4uTJ3+y8m4tLf96869x
J/qROifXa5iXYmFpTC2s9Oj7OGwWt6cxWzIOO6G+2xaUxIFn9F/jTVe3k11fFpesjLPU26zU35EtWRKQkY5vOYe+g93aYHnBZnZC
HLev+peYiZeVqrhrm2GlPKSevtT1yUo5FHYTjBd7GcrslJGj5xYO2PxbnAn9JUuELPvOK87XKCsheE/2soplGPK0FUZ1WC+mcbLw
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
""",
    "lion": """
/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAkGBwgHBgkIBwgKCgkLDRYPDQwMDRsUFRAWIB0iIiAdHx8kKDQsJCYxJx8fLT0tMTU3
Ojo6Iys/RD84QzQ5Ojf/wAALCAHWAakBAREA/8QAHAAAAQUBAQEAAAAAAAAAAAAABgABAgQFAwcI/8QARRAAAgEDAgQEBAMGBQIF
BAIDAQIDAAQRBSEGEjFBEyJRYRQycYGRobEHFSNCwdEzUmLh8CRyFjRDgvElU2NzF7KSouL/2gAIAQEAAD8AyyM0gmTXaKPerKqK
lTbmpotdKVNSpU9KkKelSNNSpqVLNLNNSpqVOCexxRVwxxU9my2l+3PB0Vu6f7V6CrxyxhlIZWGRjoRQnxLwlHOHutMUJL1eEdH+
npQC6NGxRwVYHBB9aalT0qVKmNWdMhFxqNvEejSAV64VVRDAg2RwD7YGaAf2hSBtajQHPJCP1oXFPTimO9QZaj0p6iwri4qtIK54
qyFqQWuyLiulMN66KtS6UqVNSpU9Knpx0pUs1GlTUqVMaVKlSpqVEPDXEs2ly+FOTJasdx3X6V6RbXMVzbrPbyK8TDIZaxOJeG4t
WjNxbqqXarswGz+xrza4gltpmhnQpIvVTUKVKlTZpZrtZTm3vIZV6o4NemSa7YW9qt09wjcys/Lndj0xXm+q3zajfzXTknnbIHoK
qA09OKel1qLLUDtSqDrtVWVcVyxVkDFSUZrqBUqmoqdNTUqalT09KlT5p+1NTU1KmpqVKlSpUqVKtvhzX5dHl5WJe3Y+ZCdj9Pev
SbW6juYUubVxJA4zt+dZ/EPD9vrFvkKqXIGUkHf2NeY3dtPZztDcRlJFOCDXLNNSpUqlGpeRUA3ZgBW9xikdrNZWcXNmO3XnBYnG
f0oe70qenp+1OKliubLUKR6VXmWuGDXWuqCulTQV0pVGlTU1PT0qekKelmketMaampqYmmzT0qVOKVKlSra4Z16XR7vDnmtH/wAR
OuPce9enQzRSQrNE4khYAgisziLQ4dWts4AnQfw2Xv7V5hd2slncGCYeYHB9640qakTWhw/AbnWbSPt4gNdOKboXev3kinyh+Rfb
G1ZdNmnpxUh0pCpiljNQda51CRcrXDlqaiuoG1TRa6dKVMaY0qamNLI/rUqVPT0qeketRpjTGmNMTTZpU9KnpUqVKlRNwjxGdOmF
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
UqempUq//9k=
""",
    "silhouette": """
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
hmXc7/Wq+667Vrq0wYlL/yhatP76esTK6+476S7Yg93n1LdCOcHvKbOZeGZx36uPrJOJtrjtJCnmfGpb44WjtIWc6lIIfLburftu
fPq+pUIG7a3rHlJ1fcW+cIPwsz5XGnApAIh85hnRLRYbgPBg6cEbSW/KBbcQTPlvzynC8hupXQDQHQBTcF1Z/ttzZ8J9jwZCON1q
P5092CEex32hEHPtqjsH1ADQKHOf4FTc8wwAsr0x8Z/rNjvKFWzeHgIxv9k49R1gJh1U92nmuqL7kqX03jPNp2+Yd8uhHikPV373
sLP85lNRTwm//+S7L67y9JyZ/VKnr/LMigSp2XOGVdhy+dzuo7X/7Fx+4t2ZN6Sp+KM9tfe3dWam1pE06TYok5m311vkWx98+N1T
9b9Y9Zs1Tk9Hzvfu5w+L33Sqb1y3762DgTNeo+j8yoPd78z3BA/fGy9mntMr5j5023X3v0aKOhFq6Q4Frzv37MEuGxcIdEmg9NJs
ZebktXtmxXruba7f84VopqqryPFZ/gXHT0a9+Pr8WHEGZk57J3lvuTN32Qtns+mKtGhufDIRU2T3leVvuepD337oc//e0n5WTK0r
cfIakTUjQdnprV/vKXu9OBLNPvpw7WwTPxFB7o07qz7RiE1P+IH8PlyX/dp6fNOiivw7HmpsS5JZbmpE/cfWP14ERaZ87uCkefqS
49eT0qAqveHFLz/tvFvXPPvc0gPzK3jGKtslOvuO2X5RUeLx9JIK4uFzT/d6ru7xlARnzi5qXdz83zL53sdKMoGw64aWr/39Ti09
vXxK7DF5NX7SQZJ1sZt6DMzxZ75a1/XgxuIrf3yFig7VFYwBnNEqwgHCFzr2U188HHfaw7mW9NLQhtu2rn0gBgSnajFNOn2RAmCl
k6uevuevHYfMQ9NGvf3ezeijAhjcYGVkpCSwuNeOHw3FaJ2Yt67r0m//Y6tD2F4+xfPwJk9pQaHTWqnNyfagfejRlIOT9AsdoYmI
drTS3bulDcGYqz7wgOeJ4l29dgK8fMot75O3PSrtBIVWX7GT17Rru2tDxd9/4u8TCf+wPMFI2X7uuedtB6dc9aHFj7v+ZNuN99sA
tTDV+prJU9pxh4UXRU/G7Xdag7b9neKNezbe3O6f+CUURISqPxva2Lf2rsS5LW2OxjT6MiYtPSz4/IHiw9d1OKlSdrrokLL3f25L
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
""",
}

if __name__ == "__main__":
    main()
