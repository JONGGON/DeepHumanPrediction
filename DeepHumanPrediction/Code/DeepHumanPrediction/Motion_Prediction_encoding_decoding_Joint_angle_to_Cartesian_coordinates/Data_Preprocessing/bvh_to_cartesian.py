from skeleton import process_bvhfile

import glob
import os
import json
import numpy as np
from collections import OrderedDict

last_desc = ""
files = glob.glob("Data/ACCAD/Transform_Male1_bvh/Short_data/*.bvh")
print("Number of data : {}".format(len(files)))

for fi in files:
    try:
        print("Processing %s" % fi)
        skel = process_bvhfile(fi)
    except ValueError:
        print("Skipping %s" % fi)
        continue

    body = OrderedDict()

    for frame in range(skel.frames):

        skel.create_edges_onet(frame)

        for edge in skel.edges[frame]:
            for vert in (edge.wv1, edge.wv2):
                if vert.descr is not last_desc:
                    if vert.descr not in body:
                        body[vert.descr] = []

                    body[vert.descr].append(list(vert.tr[:3]))
                    last_desc = vert.descr

    fi_name = "%s.json" % os.path.splitext(os.path.basename(fi))[0]
    if not os.path.exists("cartesian_coordinates"):
        os.makedirs("cartesian_coordinates")
    with open(os.path.join("cartesian_coordinates/{}".format(fi_name)), "w") as out_fi:
        json.dump(body, out_fi)

print("completed")
