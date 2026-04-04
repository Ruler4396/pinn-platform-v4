"""FreeFEM++ helpers for contraction_2d CFD generation."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .contraction_cases import ContractionCase


def freefem_executable() -> str:
    for candidate in ("FreeFem++", "FreeFEM++"):
        path = shutil.which(candidate)
        if path:
            return path
    raise FileNotFoundError("FreeFem++ executable not found in PATH")


def render_contraction_stokes_edp(case: ContractionCase, raw_csv_path: Path) -> str:
    beta = case.beta
    lin = case.l_in_over_w
    lc = case.lc_over_w
    lout = case.l_out_over_w
    ltot = case.total_length_over_w

    return f'''// Auto-generated contraction_2d Stokes solve for {case.case_id}
// Dimensionless geometry and fields (star variables)
// Inlet average velocity = 1, outlet pressure = 0

real beta = {beta:.12g};
real Lin = {lin:.12g};
real Lc = {lc:.12g};
real Lout = {lout:.12g};
real Ltot = {ltot:.12g};

int labIn = 1;
int labOut = 2;
int labWall = 3;

func real smoothstep5(real s) {{
  return 6.0*s^5 - 15.0*s^4 + 10.0*s^3;
}}

func real channelWidth(real xx) {{
  if (xx <= Lin) return 1.0;
  if (xx >= Lin + Lc) return beta;
  real s = (xx - Lin) / Lc;
  return 1.0 - (1.0 - beta) * smoothstep5(s);
}}

func real yTop(real xx) {{ return 0.5 * channelWidth(xx); }}
func real yBot(real xx) {{ return -0.5 * channelWidth(xx); }}

border bottomWall(t=0, 1) {{
  x = Ltot * t;
  y = yBot(x);
  label = labWall;
}}

border outletEdge(t=0, 1) {{
  x = Ltot;
  y = yBot(Ltot) + (yTop(Ltot) - yBot(Ltot)) * t;
  label = labOut;
}}

border topWall(t=0, 1) {{
  x = Ltot * (1.0 - t);
  y = yTop(x);
  label = labWall;
}}

border inletEdge(t=0, 1) {{
  x = 0.0;
  y = yTop(0.0) + (yBot(0.0) - yTop(0.0)) * t;
  label = labIn;
}}

mesh Th = buildmesh(bottomWall(180) + outletEdge(28) + topWall(180) + inletEdge(40));

fespace Vh(Th, P2);
fespace Qh(Th, P1);
Vh u, v, ut, vt;
Qh p, qt;

func real uIn(real yy) {{
  real eta = 2.0 * yy;
  return 1.5 * (1.0 - eta*eta);
}}

solve Stokes([u,v,p],[ut,vt,qt], solver=UMFPACK) =
    int2d(Th)(
      dx(u)*dx(ut) + dy(u)*dy(ut)
    + dx(v)*dx(vt) + dy(v)*dy(vt)
    - p*(dx(ut) + dy(vt))
    - qt*(dx(u) + dy(v))
    - 1.0e-10*p*qt
    )
  + on(labIn, u=uIn(y), v=0)
  + on(labWall, u=0, v=0)
  + on(labOut, p=0)
  ;

int[int] vTag(Th.nv);
for (int i = 0; i < Th.nv; ++i) vTag[i] = 0;
for (int be = 0; be < Th.nbe; ++be) {{
  int iv0 = Th.be(be)[0];
  int iv1 = Th.be(be)[1];
  int lab = Th.be(be).label;
  if (vTag[iv0] == 0 || ((lab == 1 || lab == 2) && vTag[iv0] == 3)) vTag[iv0] = lab;
  if (vTag[iv1] == 0 || ((lab == 1 || lab == 2) && vTag[iv1] == 3)) vTag[iv1] = lab;
}}

ofstream fout("{raw_csv_path.as_posix()}");
fout << "x_star,y_star,u_star,v_star,p_star,bc_tag" << endl;
for (int i = 0; i < Th.nv; ++i) {{
  real xx = Th(i).x;
  real yy = Th(i).y;
  fout << xx << "," << yy << "," << u(xx,yy) << "," << v(xx,yy) << "," << p(xx,yy) << "," << vTag[i] << endl;
}}

cout << "Saved: {raw_csv_path.as_posix()}" << endl;
'''


def run_freefem(case: ContractionCase, output_dir: Path, max_retries: int = 1) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_csv = output_dir / f"{case.case_id}_raw.csv"
    edp_path = output_dir / f"{case.case_id}_stokes.edp"
    edp_path.write_text(render_contraction_stokes_edp(case, raw_csv), encoding="utf-8")

    exe = freefem_executable()
    attempts = max(1, int(max_retries))
    last_error: subprocess.CalledProcessError | None = None
    for _ in range(attempts):
        try:
            subprocess.run([exe, "-nw", str(edp_path)], check=True, cwd=str(output_dir))
            return edp_path, raw_csv
        except subprocess.CalledProcessError as exc:
            last_error = exc
    assert last_error is not None
    raise last_error
