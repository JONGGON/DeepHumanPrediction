# BVHplay is copyright (c) 2008 Bruce Hahne.
#
# BVHplay is usable as open source software under the terms of version
# 3.0 of the Gnu Public License, which is available at
# www.gnu.org/licenses/gpl.html
#
# The author of BVHplay can be reached at hahne@io.com

import numpy as np


class WorldVert(object):
    def __init__(self, x=0, y=0, z=0, description=''):
        self.tr = np.array([x, y, z, 1])  # tr = "translate position"
        self.descr = description

    def __repr__(self):
        return "worldvert %s\n tr: %s" % (self.tr.__repr__(), self.descr)


class WorldEdge(object):
    def __init__(self, wv1, wv2, description=''):
        self.wv1 = wv1
        self.wv2 = wv2
        self.descr = description

    def __repr__(self):
        return "Worldedge %s\n wv1: %s\n wv2:%s" % (self.descr, self.wv1.__repr__(), self.wv2.__repr__())
