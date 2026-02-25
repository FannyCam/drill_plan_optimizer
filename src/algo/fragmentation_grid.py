import matplotlib.path as mpath
import numpy as np
from shapely import vectorized

from algo.constants import *
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

    def get_mask_dist_from_bootlegs(self, holes: np.ndarray) -> np.ndarray:
        """
        Génère un masque booléen pour filtrer les trous respectant la distance de sécurité.

        Cette fonction utilise le "broadcasting" NumPy pour calculer efficacement la distance
        entre chaque trou de forage et chaque zone de bootleg (zones de danger).

        Args:
            holes (np.ndarray): Matrice des coordonnées des trous (n_holes, 2).

        Returns:
            np.ndarray: Un tableau de booléens (mask) de taille (n_holes,) où True 
                        indique que le trou est à une distance sécuritaire de tous les bootlegs.
        """
        diff = holes[:, None, :] - self.bootlegs[None, :, :]
        dist2 = np.sum(diff**2, axis=2)
        mask_dist = np.all(dist2 >= BOOTLEG_DIST**2, axis=1)
        return mask_dist

    def get_hole_contributions(self, hole_x: float, hole_y: float) -> np.ndarray:
        """
        Calcule la contribution énergétique (fragmentation) d'un trou de forage sur la grille.

        Le calcul repose sur la somme de deux fonctions Gaussiennes 2D centrées sur le trou,
        représentant l'influence du tir selon deux échelles ou types d'impact (I1 et I2).

        Args:
            hole_x (float): Coordonnée X du centre du trou de forage.
            hole_y (float): Coordonnée Y du centre du trou de forage.

        Returns:
            np.ndarray: Une matrice (mesh_x.shape) représentant l'intensité de la 
                        fragmentation ajoutée par ce trou sur chaque point de la grille.
        """
        f_frag = np.zeros_like(self.mesh_x)
        I1 = A * np.exp(-(((self.mesh_x - hole_x)**2 / (2*self.sigma_x1**2)) + ((self.mesh_y - hole_y)**2 / (2*self.sigma_y1**2))))

        I2 = A * np.exp(-(((self.mesh_x - hole_x)**2 / (2*self.sigma_x2**2)) + ((self.mesh_y - hole_y)**2 / (2*self.sigma_y2**2))))

        f_frag = (I1 + I2)

        return f_frag
    
    def evaluate_fragmentation_metrics(self, holes: np.ndarray):
        """
        Évalue la qualité du plan de forage en calculant les surfaces de fragmentation.

        Cette méthode agrège les contributions de chaque trou et calcule deux métriques 
        clés basées sur des seuils d'énergie (T et T_OVER).

        Args:
            holes (np.ndarray): Liste des coordonnées (x, y) des trous de forage.

        Returns:
            tuple (float, float): 
                - A_frag : Surface totale (en m²) où la fragmentation est jugée optimale (f_frag > T).
                - A_frag_over : Surface totale (en m²) où la fragmentation est excessive (f_frag >= T_OVER).
        """
        f_frag = np.zeros_like(self.mesh_x)
        for hole_x, hole_y in holes:
            f_frag += self.get_hole_contributions(hole_x, hole_y)

        f_frag[~self.mask] = 0

        A_frag = ((f_frag > T).sum())*RESOLUTION**2
        A_frag_over = ((f_frag >= T_OVER).sum())*RESOLUTION**2

        return A_frag, A_frag_over