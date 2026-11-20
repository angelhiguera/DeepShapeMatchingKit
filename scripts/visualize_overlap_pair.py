"""Build a self-contained HTML viewer for an EchoMatch directional diagnostic."""

import argparse
import csv
import json
import math
from pathlib import Path


def read_off(path):
    """Return OFF vertices and faces, with the original zero-based indices."""
    with path.open(encoding="utf-8") as file:
        lines = (line.split("#", 1)[0].split() for line in file)
        records = (fields for fields in lines if fields)
        try:
            header = next(records)
            if header[0] != "OFF":
                raise ValueError("expected an OFF header")
            counts = header[1:] or next(records)
            vertex_count, face_count = map(int, counts[:2])
            if vertex_count <= 0 or face_count < 0:
                raise ValueError("invalid mesh counts")

            vertices = []
            for _ in range(vertex_count):
                vertex = list(map(float, next(records)[:3]))
                if len(vertex) != 3 or not all(map(math.isfinite, vertex)):
                    raise ValueError("invalid vertex coordinates")
                vertices.append(vertex)

            faces = []
            for _ in range(face_count):
                fields = next(records)
                size = int(fields[0])
                face = list(map(int, fields[1:size + 1]))
                if size < 3 or len(face) != size or any(
                    index < 0 or index >= vertex_count for index in face
                ):
                    raise ValueError("invalid face indices")
                faces.append(face)
        except (StopIteration, ValueError) as error:
            raise ValueError(f"{path}: invalid OFF mesh ({error or 'unexpected end of file'})") from error

    return {"name": path.stem, "vertices": vertices, "faces": faces}


def read_diagnostics(path, source_count, target_count):
    """Align scores, errors and target vertex IDs to the source mesh."""
    required = {"vertex_y", "overlap_probability", "gt_overlap", "target_gt", "target_fmap"}
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        fields = reader.fieldnames or ()
        missing = required.difference(fields)
        if missing:
            raise ValueError(f"{path}: missing columns: {', '.join(sorted(missing))}")
        variants = [field.removeprefix("target_") for field in fields
                    if field.startswith("target_") and field.removeprefix("target_") in fields]
        if "fmap" not in variants:
            raise ValueError(f"{path}: missing fmap errors")
        vertices = [None] * source_count
        for row in reader:
            try:
                index = int(row["vertex_y"])
                if not 0 <= index < source_count or vertices[index] is not None:
                    raise ValueError("unexpected or repeated source index")
                probability = float(row["overlap_probability"])
                gt_overlap = int(row["gt_overlap"])
                if not 0 <= probability <= 1 or gt_overlap not in (0, 1):
                    raise ValueError("expected a probability and a binary GT label")
                errors = {}
                targets = {}
                for name in variants:
                    value = float(row[name]) if row[name] else None
                    errors[name] = value if value is not None and math.isfinite(value) else None
                for name in ["gt", *variants]:
                    value = int(row[f"target_{name}"]) if row[f"target_{name}"] else None
                    if value is not None and not 0 <= value < target_count:
                        raise ValueError(f"target_{name} index outside the target mesh")
                    targets[name] = value
                vertices[index] = {
                    "probability": probability, "gtOverlap": gt_overlap,
                    "errors": errors, "targets": targets,
                }
            except (TypeError, ValueError) as error:
                raise ValueError(f"{path}:{reader.line_num}: {error}") from error
        if any(vertex is None for vertex in vertices):
            raise ValueError(f"{path}: expected one row for each of the {source_count} source vertices")
    return variants, vertices


HTML = r"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Correspondencias EchoMatch</title>
<style>
* { box-sizing: border-box; }
body { margin: auto; max-width: 1320px; padding: 24px; background: #f8fafc; color: #172338;
       font: 15px/1.5 system-ui, sans-serif; }
h1 { font-size: clamp(1.25rem, 3vw, 1.9rem); margin: 0 0 .4rem; overflow-wrap: anywhere; }
p { margin: .4rem 0 .8rem; }
.controls { display: flex; flex-wrap: wrap; align-items: end; gap: .75rem; margin-block: 1.25rem; }
label { display: grid; gap: .18rem; }
select, input, button { padding: .4rem .55rem; min-height: 38px; font: inherit; color: inherit;
                        border: 1px solid #aab6c6; border-radius: 5px; background: white; }
input { width: 8rem; }
button { cursor: pointer; }
.views { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; }
figure { margin: 0; border: 1px solid #cbd5e1; border-radius: 8px; overflow: hidden; background: white; }
canvas { display: block; width: 100%; aspect-ratio: 1.2; touch-action: none; cursor: grab; }
canvas:active { cursor: grabbing; }
figcaption { padding: .6rem .8rem; border-bottom: 1px solid #d9e1eb; font-weight: 600; }
.key { display: flex; flex-wrap: wrap; gap: .25rem 1rem; padding: .45rem .8rem; border-top: 1px solid #d9e1eb; }
.key span::before { content: ""; display: inline-block; width: .72rem; height: .72rem; border-radius: 50%;
                    margin-right: .38rem; background: var(--marker); border: 1px solid #314158; }
#details { display: grid; grid-template-columns: max-content minmax(0, 1fr); gap: .3rem .9rem; }
dt { font-weight: 600; } dd { margin: 0; font-variant-numeric: tabular-nums; }
@media (max-width: 640px) { .views { grid-template-columns: 1fr; } body { padding: 12px; } }
</style>
</head>
<body>
<h1 id="title">Correspondencias EchoMatch</h1>
<p>Arrastra para girar cada malla. Haz clic en Y o escribe el índice del vértice.</p>
<div class="controls">
  <label>Variante <select id="variant"></select></label>
  <label>Color en Y <select id="color">
    <option value="delta">Cambio de error frente a fmap</option>
    <option value="probability">Probabilidad de solapamiento</option>
    <option value="gt">Solapamiento GT</option>
    <option value="error">Error geodésico</option>
  </select></label>
  <label>Vértice de Y <input id="vertex" type="number" min="0" step="1"></label>
  <button type="button" id="worst">Ir al mayor aumento de error</button>
</div>
<div class="views">
  <figure>
    <figcaption id="source-title">Origen Y</figcaption>
    <canvas id="source" aria-label="Malla de origen Y"></canvas>
    <div class="key">
      <span style="--marker:#f9c033">Vértice elegido</span>
      <span style="--marker:#08a69b">Vecinos por arista</span>
    </div>
  </figure>
  <figure>
    <figcaption id="target-title">Destino X</figcaption>
    <canvas id="target" aria-label="Malla de destino X"></canvas>
    <div class="key">
      <span style="--marker:#f9c033">Destino predicho</span>
      <span style="--marker:#8770d6">Destino GT</span>
      <span style="--marker:#08a69b">Destinos predichos de los vecinos</span>
    </div>
  </figure>
</div>
<p id="scale"></p>
<dl id="details" aria-live="polite"></dl>

<script type="application/json" id="mesh-data">__DATA__</script>
<script>
"use strict";
const data = JSON.parse(document.querySelector("#mesh-data").textContent);
const get = id => document.getElementById(id);
const variant = get("variant");
const colorMode = get("color");
const vertex = get("vertex");
const details = get("details");
const gold = "#f9c033", teal = "#08a69b", violet = "#8770d6";
get("title").textContent = `${data.source.name} → ${data.target.name}`;
get("source-title").textContent = `Origen Y · ${data.source.name}`;
get("target-title").textContent = `Destino X · ${data.target.name}`;

for (const name of data.variants) {
  variant.add(new Option(name, name));
}
if (data.variants.includes("fmap_predicted")) {
  variant.value = "fmap_predicted";
}
vertex.max = data.vertices.length - 1;

const adjacency = data.vertices.map(() => new Set());
for (const face of data.source.faces) {
  for (let i = 0; i < face.length; i++) {
    const current = face[i], next = face[(i + 1) % face.length];
    adjacency[current].add(next);
    adjacency[next].add(current);
  }
}

function createView(id, mesh, pickVertex) {
  const canvas = get(id);
  const ctx = canvas.getContext("2d");
  const minimum = [Infinity, Infinity, Infinity], maximum = [-Infinity, -Infinity, -Infinity];
  for (const point of mesh.vertices) {
    for (let axis = 0; axis < 3; axis++) {
      minimum[axis] = Math.min(minimum[axis], point[axis]);
      maximum[axis] = Math.max(maximum[axis], point[axis]);
    }
  }
  const center = minimum.map((value, axis) => (value + maximum[axis]) / 2);
  const span = Math.max(...maximum.map((value, axis) => value - minimum[axis])) || 1;
  const points = mesh.vertices.map(point => point.map((value, axis) => (value - center[axis]) / span));
  let yaw = -.35, pitch = .15, projected = [], drag = null;

  function draw(fill, markerSets) {
    const width = canvas.clientWidth, height = canvas.clientHeight;
    if (!width || !height) return;
    const ratio = window.devicePixelRatio || 1;
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.fillStyle = "#eef3f8";
    ctx.fillRect(0, 0, width, height);
    const factor = Math.min(width, height) * .70;
    const cosY = Math.cos(yaw), sinY = Math.sin(yaw);
    const cosP = Math.cos(pitch), sinP = Math.sin(pitch);
    projected = points.map(([x, y, z]) => {
      const u = cosY * x + sinY * z;
      const w = cosY * z - sinY * x;
      return [width / 2 + factor * u, height / 2 - factor * (cosP * y - sinP * w),
              sinP * y + cosP * w];
    });

    const sortedFaces = mesh.faces.map(face => ({
      face, distance: face.reduce((sum, index) => sum + projected[index][2], 0) / face.length
    })).sort((a, b) => a.distance - b.distance);
    ctx.strokeStyle = "#667b93";
    ctx.lineWidth = .26;
    for (const {face} of sortedFaces) {
      ctx.beginPath();
      ctx.moveTo(projected[face[0]][0], projected[face[0]][1]);
      for (const index of face.slice(1)) ctx.lineTo(projected[index][0], projected[index][1]);
      ctx.closePath();
      ctx.fillStyle = fill(face);
      ctx.fill();
      ctx.stroke();
    }

    for (const [indices, radius, color, outline] of markerSets) {
      ctx.fillStyle = color;
      ctx.strokeStyle = outline;
      ctx.lineWidth = 2;
      for (const index of new Set(indices)) {
        if (index === null || index === undefined) continue;
        const [x, y] = projected[index];
        ctx.beginPath();
        ctx.arc(x, y, radius, 0, 2 * Math.PI);
        ctx.fill();
        ctx.stroke();
      }
    }
  }

  canvas.addEventListener("pointerdown", event => {
    drag = {x: event.clientX, y: event.clientY, distance: 0};
    canvas.setPointerCapture(event.pointerId);
  });
  canvas.addEventListener("pointermove", event => {
    if (!drag) return;
    const dx = event.clientX - drag.x, dy = event.clientY - drag.y;
    drag.distance += Math.abs(dx) + Math.abs(dy);
    drag.x = event.clientX;
    drag.y = event.clientY;
    yaw += dx * .012;
    pitch += dy * .012;
    render();
  });
  canvas.addEventListener("pointerup", event => {
    if (drag && drag.distance < 5 && pickVertex) {
      const bounds = canvas.getBoundingClientRect();
      const x = event.clientX - bounds.left, y = event.clientY - bounds.top;
      let nearest = null, minimum = 14 ** 2, depth = -Infinity;
      for (let index = 0; index < projected.length; index++) {
        const [u, v, z] = projected[index];
        const distance = (x - u) ** 2 + (y - v) ** 2;
        if (distance < minimum || (distance === minimum && z > depth)) {
          [nearest, minimum, depth] = [index, distance, z];
        }
      }
      if (nearest !== null) pickVertex(nearest);
    }
    drag = null;
  });
  canvas.addEventListener("pointercancel", () => { drag = null; });
  return draw;
}

function delta(row, name) {
  return row.errors[name] === null || row.errors.fmap === null
    ? null : row.errors[name] - row.errors.fmap;
}

function colorBetween(a, b, fraction) {
  const rgb = a.map((value, index) => Math.round(value + fraction * (b[index] - value)));
  return `rgb(${rgb.join(",")})`;
}

function number(value) {
  return value === null ? "—" : value.toFixed(4);
}

function select(index) {
  vertex.value = index;
  render();
}

function selectWorst() {
  let index = 0, maximum = -Infinity;
  for (const [candidate, row] of data.vertices.entries()) {
    const value = delta(row, variant.value);
    if (value !== null && value > maximum) {
      [index, maximum] = [candidate, value];
    }
  }
  select(index);
}

const sourceView = createView("source", data.source, select);
const targetView = createView("target", data.target, null);

function render() {
  const index = vertex.valueAsNumber;
  if (!Number.isInteger(index) || index < 0 || index >= data.vertices.length) return;
  const row = data.vertices[index], neighbors = [...adjacency[index]], name = variant.value;
  const mode = colorMode.value;
  const values = data.vertices.map(item => {
    if (mode === "delta") return delta(item, name);
    if (mode === "error") return item.errors[name];
    return mode === "gt" ? item.gtOverlap : item.probability;
  });
  const maximum = mode === "error" || mode === "delta"
    ? (values.reduce((max, value) => value === null ? max : Math.max(max, Math.abs(value)), 0) || 1)
    : 1;
  const modeLabels = {
    delta: `Azul ${number(-maximum)} · blanco 0 · rojo +${number(maximum)}`,
    error: `Blanco 0 · rojo ${number(maximum)}`,
    gt: "Blanco 0 · verde 1",
    probability: "Blanco 0 · verde 1",
  };
  get("scale").textContent = `${modeLabels[mode]}. Gris sin dato. Cada cara promedia sus vértices.`;

  const neutral = [241, 246, 251], blue = [40, 103, 183], red = [185, 41, 54], green = [0, 132, 108];
  function sourceColor(face) {
    const valid = face.map(i => values[i]).filter(value => value !== null);
    if (!valid.length) return "#a7b7ca";
    const mean = valid.reduce((total, value) => total + value, 0) / valid.length;
    const end = mode === "delta" ? (mean < 0 ? blue : red) : mode === "error" ? red : green;
    return colorBetween(neutral, end, Math.min(1, Math.abs(mean) / maximum));
  }

  sourceView(sourceColor, [[neighbors, 4, teal, "#154b54"], [[index], 7, gold, "#644910"]]);
  const groundTruth = row.targets.gt, prediction = row.targets[name];
  targetView(() => "#e0e8f1", [
    [neighbors.map(i => data.vertices[i].targets[name]), 4, teal, "#154b54"],
    [[groundTruth], 9, violet, "#3d2b72"],
    [[prediction], 6, gold, "#644910"],
  ]);

  const neighborMean = neighbors.length
    ? neighbors.reduce((total, i) => total + data.vertices[i].probability, 0) / neighbors.length
    : null;
  const neighborTargets = new Set(neighbors.map(i => data.vertices[i].targets[name])
                                            .filter(index => index !== null));
  const stats = [
    ["Vértice Y", index],
    ["Probabilidad / GT", `${number(row.probability)} / ${row.gtOverlap}`],
    ["Vecinos por arista", `${neighbors.length} · probabilidad media ${number(neighborMean)}`],
    ["Destinos únicos de los vecinos", neighborTargets.size],
    ["Vértices X predicho / GT", `${prediction ?? "—"} / ${groundTruth ?? "—"}`],
    ["Error variante / fmap", `${number(row.errors[name])} / ${number(row.errors.fmap)}`],
    ["Cambio frente a fmap", number(delta(row, name))],
  ];
  details.replaceChildren();
  for (const [label, value] of stats) {
    const term = document.createElement("dt"), description = document.createElement("dd");
    term.textContent = label;
    description.textContent = value;
    details.append(term, description);
  }
}

get("worst").addEventListener("click", selectWorst);
for (const control of [variant, colorMode, vertex]) control.addEventListener("input", render);
window.addEventListener("resize", render);
selectWorst();
</script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True, help="Directional vertex diagnostic CSV")
    parser.add_argument("--source", type=Path, required=True, help="Source Y OFF mesh")
    parser.add_argument("--target", type=Path, required=True, help="Target X OFF mesh")
    parser.add_argument("--output", type=Path, required=True, help="Destination HTML file")
    args = parser.parse_args()
    try:
        source, target = read_off(args.source), read_off(args.target)
        variants, vertices = read_diagnostics(
            args.csv, len(source["vertices"]), len(target["vertices"])
        )
        data = {"source": source, "target": target, "variants": variants, "vertices": vertices}
        payload = json.dumps(data, allow_nan=False, separators=(",", ":")).replace("<", r"\u003c")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(HTML.replace("__DATA__", payload), encoding="utf-8")
    except (OSError, ValueError) as error:
        parser.exit(1, f"{parser.prog}: {error}\n")
    print(args.output)


if __name__ == "__main__":
    main()
