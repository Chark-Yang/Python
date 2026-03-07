from pathlib import Path
import mujoco
p = Path('model/jaka_description/urdf/jaka_s5_mujoco.urdf')
print('exists:', p.exists(), '->', str(p))
if not p.exists():
    p2 = Path('model/jaka_description/urdf/jaka_s5.urdf')
    print('original exists:', p2.exists(), str(p2))
    if not p2.exists():
        raise SystemExit(1)
    text = p2.read_text()
    from pathlib import Path as _P
    repo_root = _P(__file__).resolve().parent
    meshes_dir = (repo_root / 'model' / 'jaka_description' / 'meshes').as_posix() + '/'
    text = text.replace('package://jaka_description/meshes/', meshes_dir)
    p.write_text(text)

model = mujoco.MjModel.from_xml_path(str(p))
print('Loaded model OK, ngeom=', len(model.geoms), 'njoints=', len(model.joints))
