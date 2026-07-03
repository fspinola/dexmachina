import sys, pickle
in_pkl, urdf, out_pkl = sys.argv[1], sys.argv[2], sys.argv[3]
d = pickle.load(open(in_pkl, "rb"))           # runs on a GPU node -> cuda tensors load fine
d["robot_cfgs"]["left"]["urdf_path"] = urdf   # point ONLY the left hand at the temp urdf
pickle.dump(d, open(out_pkl, "wb"))
print("patched left urdf ->", urdf)
