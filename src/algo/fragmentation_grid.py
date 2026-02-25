import numpy as np
import matplotlib.path as mpath
from algo.constants import *

from shapely import vectorized



class FragmentationGrid:
    def __init__(self, polygon, bootlegs):
        self.x_min, self.y_min, self.x_max, self.y_max = polygon.bounds
        self.bootlegs = bootlegs

        self.axe_x = np.arange(self.x_min, self.x_max + RESOLUTION, RESOLUTION)
        self.axe_y = np.arange(self.y_min, self.y_max + RESOLUTION, RESOLUTION)

        self.mesh_x, self.mesh_y = np.meshgrid(self.axe_x, self.axe_y)

        mask_in = vectorized.contains(polygon, self.mesh_x.ravel(), self.mesh_y.ravel()) | vectorized.touches(polygon, self.mesh_x.ravel(), self.mesh_y.ravel())
        self.mask = mask_in.reshape(self.mesh_x.shape)
    
        self.sigma_x1 = S / np.sqrt(8 * np.log(2*A/C))
        self.sigma_y1 = B / np.sqrt(6 * np.log(2*A/C))
        self.sigma_x2 = S / np.sqrt(32 * np.log(2*A/C))
        self.sigma_y2 = B / np.sqrt(2 * np.log(2*A/C))

    def get_mask_dist_from_bootlegs(self, holes):
        diff = holes[:, None, :] - self.bootlegs[None, :, :]
        dist2 = np.sum(diff**2, axis=2)
        mask_dist = np.all(dist2 >= BOOTLEG_DIST**2, axis=1)
        return mask_dist

    def get_hole_contributions(self, hole_x, hole_y):
        f_frag = np.zeros_like(self.mesh_x)
        I1 = A * np.exp(-(((self.mesh_x - hole_x)**2 / (2*self.sigma_x1**2)) + ((self.mesh_y - hole_y)**2 / (2*self.sigma_y1**2))))

        I2 = A * np.exp(-(((self.mesh_x - hole_x)**2 / (2*self.sigma_x2**2)) + ((self.mesh_y - hole_y)**2 / (2*self.sigma_y2**2))))

        f_frag = (I1 + I2)

        return f_frag
    
    def evaluate_fragmentation_metrics(self, holes):
        f_frag = np.zeros_like(self.mesh_x)
        for hole_x, hole_y in holes:
            f_frag += self.get_hole_contributions(hole_x, hole_y)

        f_frag[~self.mask] = 0

        A_frag = ((f_frag > T).sum())*RESOLUTION**2
        A_frag_over = ((f_frag >= T_OVER).sum())*RESOLUTION**2

        return A_frag, A_frag_over