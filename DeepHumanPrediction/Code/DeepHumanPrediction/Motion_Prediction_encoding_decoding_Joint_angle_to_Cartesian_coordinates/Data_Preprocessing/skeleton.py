# BVHplay is copyright (c) 2008 Bruce Hahne.
#
# BVHplay is usable as open source software under the terms of version
# 3.0 of the Gnu Public License, which is available at
# www.gnu.org/licenses/gpl.html
#
# The author of BVHplay can be reached at hahne@io.com

"""
AVOIDING OFF-BY-ONE ERRORS:
Let N be the total number of keyframes in the BVH file.  Then:
- bvh.keyframes[] is an array that runs from 0 to N-1
- skeleton.keyframes[] is another reference to bvh.keyframes and similarly
  runs from 0 to N-1
- skeleton.edges{t} is a dict where t can run from 1 to N
- joint.trtr{t} is a dict where t can run from 1 to N
- joint.worldpos{t} is a dict where t can run from 1 to N

So if you're talking about raw BVH keyframe rows from the file,
you use an array and the values run from 0 to N-1.  This is an artifact
of using .append to create bvh.keyframes.

By contrast, if you're talking about a non-keyframe data structure
derived from the BVH keyframes, such as matrices or edges, it's a
dictionary and the values run from 1 to N.
"""

from cgkit_bvh import BVHReader
from geo import WorldVert, WorldEdge
from numpy import dot
import numpy as np


class Joint(object):
    """
    A BVH "joint" is a single vertex with potentially MULTIPLE
    edges.  It's not accurate to call these "bones" because if
    you rotate the joint, you rotate ALL attached bones.
    """

    def __init__(self, name):
        self.name = name
        self.children = []
        self.channels = []
        # list entry is one of [XYZ]position, [XYZ]rotation
        self.hasparent = 0  # flag
        self.parent = None
        # static translation vector (x, y, z)
        self.strans = np.zeros(3)

        # Transformation matrices:
        self.stransmat = np.zeros((4, 4))

        # A premultiplied series of translation and rotation matrices
        self.trtr = {}
        # Time-based worldspace xyz position of the joint's endpoint.  A list of vec4's
        self.worldpos = {}

    def info(self):
        print("Joint name:", self.name)
        print(" %s is connected to " % self.name, )
        if len(self.children) == 0:
            print("nothing")
        else:
            for child in self.children:
                print("%s " % child.name, )
            print()
        for child in self.children:
            child.info()

    def __repr__(self):  # Recursively build up text info
        str2 = self.name + " at strans=" + str(self.strans) + " is connected to "
        if len(self.children) == 0:
            str2 = str2 + "nothing\n"
        else:
            for child in self.children:
                str2 = str2 + child.name + " "
            str2 = str2 + "\n"
        str3 = ""
        for child in self.children:
            str3 = str3 + child.__repr__()
        str1 = str2 + str3
        return str1

    def addchild(self, childjoint):
        self.children.append(childjoint)
        childjoint.hasparent = 1
        childjoint.parent = self

    def create_edges_recurse(self, edgelist, t, debug=0):
        if debug:
            print("create_edge_recurse starting for joint ", self.name)
        if self.hasparent:
            temp1 = self.parent.worldpos[t]  # Faster than triple lookup below?
            temp2 = self.worldpos[t]
            v1 = WorldVert(temp1[0], temp1[1], temp1[2], description=self.parent.name)
            v2 = WorldVert(temp2[0], temp2[1], temp2[2], description=self.name)

            descr = "%s to %s" % (self.parent.name, self.name)
            myedge = WorldEdge(v1, v2, description=descr)
            edgelist.append(myedge)

        for child in self.children:
            if debug:
                print(" Recursing for child ", child.name)
            child.create_edges_recurse(edgelist, t, debug)


class Skeleton(object):
    """
    This class is actually for a skeleton plus some time-related info
      frames: number of frames in the animation
      dt: delta-t in seconds per frame (default: 30fps i.e. 1/30)
    """

    def __init__(self, hips, keyframes, frames=0, dt=.033333333):
        self.hips = hips
        self.keyframes = keyframes
        self.frames = frames
        self.dt = dt
        self.edges = {}

        # Precompute hips min and max values in all 3 dimensions.
        # First determine how far into a keyframe we need to look to find the
        # XYZ hip positions
        offset = 0
        xoffset = 0
        yoffset = 0
        zoffset = 0

        for channel in self.hips.channels:
            if channel == "Xposition":
                xoffset = offset
            if channel == "Yposition":
                yoffset = offset
            if channel == "Zposition":
                zoffset = offset
            offset += 1

        self.minx = np.inf
        self.miny = np.inf
        self.minz = np.inf
        self.maxx = -np.inf
        self.maxy = -np.inf
        self.maxz = -np.inf
        # We can't just look at the keyframe values, we also have to correct
        # by the static hips OFFSET value, since sometimes this can be quite
        # large.  I feel it's bad BVH file form to have a non-zero HIPS offset
        # position, but there are definitely files that do this.
        xcorrect = self.hips.strans[0]
        ycorrect = self.hips.strans[1]
        zcorrect = self.hips.strans[2]

        for keyframe in self.keyframes:
            x = keyframe[xoffset] + xcorrect
            y = keyframe[yoffset] + ycorrect
            z = keyframe[zoffset] + zcorrect
            if x < self.minx:
                self.minx = x
            if x > self.maxx:
                self.maxx = x
            if y < self.miny:
                self.miny = y
            if y > self.maxy:
                self.maxy = y
            if z < self.minz:
                self.minz = z
            if z > self.maxz:
                self.maxz = z

    def __repr__(self):
        return "frames = %s, dt = %s\n%s" % (self.frames, self.dt, self.hips.__repr__())

    def create_edges_onet(self, t, debug=0):
        if debug:
            print("create_edges_onet starting for t=", t)

        # Before we can compute edge positions, we need to have called
        # process_bvhkeyframe for time t, which computes trtr and worldpos
        # for the joint hierarchy at time t.  Since we no longer precompute
        # this information when we read the BVH file, here's where we do it.
        # This is on-demand computation of trtr and worldpos.
        if t not in self.hips.worldpos:
            if debug:
                print("create_edges_onet: about to call process_bvhkeyframe for t=", t)
            process_bvhkeyframe(self.keyframes[t - 1], self.hips, t, debug=debug)

        edgelist = []
        if t not in self.edges:
            if debug:
                print("create_edges_onet: creating edges for t=", t)

            self.hips.create_edges_recurse(edgelist, t, debug=debug)
            self.edges[t] = edgelist  # dictionary entry

        if debug:
            print("create_edges edge list at timestep %d:" % t)
            print(edgelist)


class ReadBvh(BVHReader):

    def __init__(self, filename):
        super(ReadBvh, self).__init__(filename)
        self.root = None
        self.dt = None
        self.frames = None
        self.keyframes = None

    def on_hierarchy(self, root):
        self.root = root
        self.keyframes = []

    def on_motion(self, frames, dt):
        self.frames = frames
        self.dt = dt

    def on_frame(self, values):
        self.keyframes.append(values)


def process_bvhnode(node, parentname='hips'):
    """
    Recursively process a BVHReader node object and return the root joint
    of a bone hierarchy.  This routine creates a new joint hierarchy.
    It isn't a Skeleton yet since we haven't read any keyframes or
    created a Skeleton class yet.
  
    Steps:
    1. Create a new joint
    2. Copy the info from Node to the new joint
    3. For each Node child, recursively call myself
    4. Return the new joint as retval
  
    We have to pass in the parent name because this routine
    needs to be able to name the leaves "parentnameEnd" instead
    of "End Site"
    """

    name = node.name
    if (name == "End Site") or (name == "end site"):
        name = parentname + "End"

    b1 = Joint(name)
    b1.channels = node.channels
    b1.strans[0] = node.offset[0]
    b1.strans[1] = node.offset[1]
    b1.strans[2] = node.offset[2]

    # Compute static translation matrix from vec3 b1.strans
    b1.stransmat = np.eye(4)

    b1.stransmat[0, 3] = b1.strans[0]
    b1.stransmat[1, 3] = b1.strans[1]
    b1.stransmat[2, 3] = b1.strans[2]

    for child in node.children:
        b2 = process_bvhnode(child, name)  # Creates a child joint "b2"
        b1.addchild(b2)
    return b1


def process_bvhkeyframe(keyframe, joint, t, debug=0):
    """
    Recursively extract (occasionally) translation and (mostly) rotation
    values from a sequence of floats and assign to joints.

    Takes a keyframe (a list of floats) and returns a new keyframe that
    contains the not-yet-processed (not-yet-eaten) floats of the original
    sequence of floats.  Also assigns the eaten floats to the appropriate
    class variables of the appropriate Joint object.

    This function could technically be a class function within the Joint
    class, but to maintain similarity with process_bvhnode I won't do that.
    """

    counter = 0
    dotrans = 0

    xpos = 0
    ypos = 0
    zpos = 0

    xrot = 0
    yrot = 0
    zrot = 0

    # We have to build up drotmat one rotation value at a time so that
    # we get the matrix multiplication order correct.
    drotmat = np.eye(4)

    if debug:
        print(" process_bvhkeyframe: doing joint %s, t=%d" % (joint.name, t))
        print(" keyframe has %d elements in it." % (len(keyframe)))

    # Suck in as many values off the front of "keyframe" as we need
    # to populate this joint's channels.  The meanings of the keyvals
    # aren't given in the keyframe itself; their meaning is specified
    # by the channel names.
    for channel in joint.channels:
        keyval = keyframe[counter]

        if channel == "Xposition":
            dotrans = 1
            xpos = keyval

        elif channel == "Yposition":
            dotrans = 1
            ypos = keyval

        elif channel == "Zposition":
            dotrans = 1
            zpos = keyval

        elif channel == "Xrotation":
            xrot = keyval
            theta = np.radians(xrot)
            mycos = np.cos(theta)
            mysin = np.sin(theta)
            drotmat2 = np.eye(4)
            drotmat2[1, 1] = mycos
            drotmat2[1, 2] = -mysin
            drotmat2[2, 1] = mysin
            drotmat2[2, 2] = mycos
            drotmat = np.dot(drotmat, drotmat2)

        elif channel == "Yrotation":
            yrot = keyval
            theta = np.radians(yrot)
            mycos = np.cos(theta)
            mysin = np.sin(theta)
            drotmat2 = np.eye(4)
            drotmat2[0, 0] = mycos
            drotmat2[0, 2] = mysin
            drotmat2[2, 0] = -mysin
            drotmat2[2, 2] = mycos
            drotmat = np.dot(drotmat, drotmat2)

        elif channel == "Zrotation":
            zrot = keyval
            theta = np.radians(zrot)
            mycos = np.cos(theta)
            mysin = np.sin(theta)
            drotmat2 = np.eye(4)
            drotmat2[0, 0] = mycos
            drotmat2[0, 1] = -mysin
            drotmat2[1, 0] = mysin
            drotmat2[1, 1] = mycos
            drotmat = dot(drotmat, drotmat2)

        else:
            print("Fatal error in process_bvhkeyframe: illegal channel name ", channel)
            return 0

        counter += 1

    if dotrans:  # If we are the hips...
        # Build a translation matrix for this keyframe
        dtransmat = np.eye(4)
        dtransmat[0, 3] = xpos
        dtransmat[1, 3] = ypos
        dtransmat[2, 3] = zpos

        if debug:
            print("Joint %s: xpos ypos zpos is %s %s %s" % (joint.name, xpos, ypos, zpos))

        if debug:
            print("Joint %s: xrot yrot zrot is %s %s %s" % (joint.name, xrot, yrot, zrot))

        localtoworld = np.dot(joint.stransmat, dtransmat)

    else:
        parent_trtr = joint.parent.trtr[t]  # Dictionary-based rewrite

        localtoworld = np.dot(parent_trtr, joint.stransmat)

    # At this point we should have computed:
    #  stransmat  (computed previously in process_bvhnode subroutine)
    #  dtransmat (only if we're the hips)
    #  drotmat
    # We now have enough to compute joint.trtr and also to convert
    # the position of this joint (vertex) to worldspace.
    #
    # For the non-hips case, we assume that our parent joint has already
    # had its trtr matrix appended to the end of self.trtr[]
    # and that the appropriate matrix from the parent is the LAST item
    # in the parent's trtr[] matrix list.
    #
    # Worldpos of the current joint is localtoworld = TRTR...T*[0,0,0,1]
    #   which equals parent_trtr * T*[0,0,0,1]
    # In other words, the rotation value of a joint has no impact on
    # that joint's position in space, so drotmat doesn't get used to
    # compute worldpos in this routine.
    #
    # However we don't pass localtoworld down to our child -- what
    # our child needs is trtr = TRTRTR...TR
    #
    # The code below attempts to optimize the computations so that we
    # compute localtoworld first, then trtr.

    trtr = np.dot(localtoworld, drotmat)

    joint.trtr[t] = trtr  # New dictionary-based approach

    # numpy conversion: eliminate the matrix multiplication entirely,
    # since all we're doing is extracting the last column of worldpos.
    worldpos = np.array([localtoworld[0, 3], localtoworld[1, 3], localtoworld[2, 3], localtoworld[3, 3]])
    joint.worldpos[t] = worldpos

    if debug:
        print("Joint %s: here are some matrices" % joint.name)
        print("stransmat:")
        print(joint.stransmat)
        if not joint.hasparent:  # if hips
            print("   dtransmat:")
            print(dtransmat)
        print("   drotmat:")
        print(drotmat)
        print("   localtoworld:")
        print(localtoworld)
        print("   trtr:")
        print(trtr)
        print("  worldpos:", worldpos)
        print()

    newkeyframe = keyframe[counter:]
    for child in joint.children:
        # Here's the recursion call.  Each time we call process_bvhkeyframe,
        # the returned value "newkeyframe" should shrink due to the slicing
        # process
        newkeyframe = process_bvhkeyframe(newkeyframe, child, t, debug=debug)
        if newkeyframe == 0:
            print("Passing up fatal error in process_bvhkeyframe")
            return 0
    return newkeyframe


def process_bvhfile(filename, debug=0):
    """
    The caller of this routine should cover possible exceptions.
    Here are two possible errors:
     IOError: [Errno 2] No such file or directory: 'fizzball'
     raise SyntaxError, "Syntax error in line %d: 'HIERARCHY' expected, \
       got '%s' instead"%(self.linenr, tok)
    
    Here's some information about the two mybvh calls:
    
    mybvh.read() returns a readbvh instance:
     retval from readbvh() is  <skeleton.readbvh instance at 0x176dcb0>
    So this isn't useful for error-checking.
    
    mybvh.read() returns None on success and throws an exception on failure.
    """

    mybvh = ReadBvh(filename)
    retval = mybvh.read()

    hips = process_bvhnode(mybvh.root)  # Create joint hierarchy
    myskeleton = Skeleton(hips, keyframes=mybvh.keyframes, frames=mybvh.frames, dt=mybvh.dt)
    if debug:
        print("skeleton is: ", myskeleton)
    return myskeleton
