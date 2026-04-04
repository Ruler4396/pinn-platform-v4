"""FreeFEM++ helpers for bend_2d CFD generation."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .bend_cases import BendCase


def freefem_executable() -> str:
    for candidate in ("FreeFem++", "FreeFEM++"):
        path = shutil.which(candidate)
        if path:
            return path
    raise FileNotFoundError("FreeFem++ executable not found in PATH")


def inlet_profile_expr(case: BendCase, eta_symbol: str = "eta") -> str:
    eta_norm = f"({eta_symbol}/halfW)"
    if case.inlet_profile_name == "parabolic":
        return f"1.5 * (1.0 - ({eta_norm})*({eta_norm}))"
    if case.inlet_profile_name == "blunted":
        return f"1.25 * (1.0 - pow({eta_norm}, 4))"
    if case.inlet_profile_name == "skewed_top":
        return f"(1.5 * (1.0 - ({eta_norm})*({eta_norm}))) * (1.0 + 0.4*{eta_norm})"
    if case.inlet_profile_name == "skewed_bottom":
        return f"(1.5 * (1.0 - ({eta_norm})*({eta_norm}))) * (1.0 - 0.4*{eta_norm})"
    raise ValueError(f"Unsupported inlet profile '{case.inlet_profile_name}'")


def render_bend_stokes_edp(case: BendCase, raw_csv_path: Path) -> str:
    alpha = case.theta_deg * 3.141592653589793 / 180.0
    inlet_expr = inlet_profile_expr(case)
    return f'''// Auto-generated bend_2d Stokes solve for {case.case_id}
// Dimensionless geometry and fields (star variables)
// Inlet average velocity = 1, outlet pressure = 0
// Inlet profile = {case.inlet_profile_name}

real W = 1.0;
real halfW = 0.5;
real Lin = {case.l_in_over_w:.12g};
real Lout = {case.l_out_over_w:.12g};
real Rc = {case.rc_over_w:.12g};
real turnAngle = {alpha:.12g};
real ang0 = -pi/2.0;
real ang1 = ang0 + turnAngle;
real xC = Lin;
real yC = Rc;
real Ri = Rc - halfW;
real Ro = Rc + halfW;
real xArcEnd = xC + Rc*cos(ang1);
real yArcEnd = yC + Rc*sin(ang1);
real txOut = cos(turnAngle);
real tyOut = sin(turnAngle);
real nxOut = -sin(turnAngle);
real nyOut = cos(turnAngle);

int labIn = 1;
int labOut = 2;
int labWall = 3;

func real uIn(real eta) {{
  return {inlet_expr};
}}

border inletBottomWall(t=0, 1) {{
  x = Lin * t;
  y = -halfW;
  label = labWall;
}}

border outerArc(t=0, 1) {{
  real ang = ang0 + (ang1 - ang0) * t;
  x = xC + Ro*cos(ang);
  y = yC + Ro*sin(ang);
  label = labWall;
}}

border outletRightWall(t=0, 1) {{
  real xi = Lout * t;
  x = xArcEnd + xi*txOut - halfW*nxOut;
  y = yArcEnd + xi*tyOut - halfW*nyOut;
  label = labWall;
}}

border outletEdge(t=0, 1) {{
  real eta = -halfW + W * t;
  x = xArcEnd + Lout*txOut + eta*nxOut;
  y = yArcEnd + Lout*tyOut + eta*nyOut;
  label = labOut;
}}

border outletLeftWall(t=0, 1) {{
  real xi = Lout * (1.0 - t);
  x = xArcEnd + xi*txOut + halfW*nxOut;
  y = yArcEnd + xi*tyOut + halfW*nyOut;
  label = labWall;
}}

border innerArc(t=0, 1) {{
  real ang = ang1 + (ang0 - ang1) * t;
  x = xC + Ri*cos(ang);
  y = yC + Ri*sin(ang);
  label = labWall;
}}

border inletTopWall(t=0, 1) {{
  x = Lin * (1.0 - t);
  y = halfW;
  label = labWall;
}}

border inletEdge(t=0, 1) {{
  real eta = halfW - W * t;
  x = 0.0;
  y = eta;
  label = labIn;
}}

int nIn = max(48, int(22 * Lin));
int nArc = max(72, int(22 * Rc * turnAngle));
int nOut = max(64, int(22 * Lout));
int nEdge = 36;
mesh Th = buildmesh(
    inletBottomWall(nIn)
  + outerArc(nArc)
  + outletRightWall(nOut)
  + outletEdge(nEdge)
  + outletLeftWall(nOut)
  + innerArc(nArc)
  + inletTopWall(nIn)
  + inletEdge(nEdge)
);

fespace Vh(Th, P2);
fespace Qh(Th, P1);
Vh u, v, ut, vt;
Qh p, qt;

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



def run_freefem(case: BendCase, output_dir: Path, max_retries: int = 1) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_csv = output_dir / f"{case.case_id}_raw.csv"
    edp_path = output_dir / f"{case.case_id}_stokes.edp"
    edp_path.write_text(render_bend_stokes_edp(case, raw_csv), encoding="utf-8")

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
