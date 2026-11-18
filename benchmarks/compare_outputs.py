"""Compare two output dirs: voxel identity + affine agreement (nibabel), per case.
python benchmarks/compare_outputs.py <ref_dir> <test_dir>"""
import glob, sys, os
import nibabel as nib, numpy as np
ref, tst = sys.argv[1], sys.argv[2]
n = same = aff_ok = 0
for f in sorted(glob.glob(ref + "/*.nii.gz")):
    g = os.path.join(tst, os.path.basename(f))
    if not os.path.exists(g):
        print("missing", g); continue
    a, b = nib.load(f), nib.load(g); n += 1
    x, y = np.asarray(a.dataobj), np.asarray(b.dataobj)
    s = x.shape == y.shape and np.array_equal(x, y); same += s
    ok = np.allclose(a.affine, b.affine, atol=1e-4); aff_ok += ok
    if not (s and ok):
        print(os.path.basename(f), "voxels_equal", s, "affine_close", ok, "max|daff|", np.abs(a.affine - b.affine).max())
print(f"{n} cases: voxel-identical {same}, affine-equal {aff_ok}")
