import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from shapely import vectorized
from shapely.geometry import Polygon

from algo.constants import *
from algo.fragmentation_grid import FragmentationGrid

class DrillModel:
    def __init__(self, boundary, bootlegs):
        self.boundary = boundary
        self.bootlegs = bootlegs
        self.polygon = Polygon(self.boundary)
        self.fragmentation_grid = FragmentationGrid(self.polygon, self.bootlegs)
        self.current_holes = np.empty((0, 2))
        self.current_holes_non_filtered = np.empty((0, 2))
        self.current_objective = 0
        self.current_dx = 0
        self.current_dy = 0

    def optimize(self):
        """
        Exécute le processus complet d'optimisation du plan de forage.

        L'algorithme procède en deux phases :
        1. Heuristique initiale : Génère une première solution viable.
        2. Recuit Simulé : Affine la solution en explorant l'espace des possibles pour 
           maximiser la fragmentation tout en minimisant la sur-fragmentation.
        """
        self.determine_initial_solution()
        self.improve_solution_by_simulated_annealing()

    def get_objective(self, holes: np.ndarray) -> float:
        """
        Calcule le score de performance global (fitness) d'un plan de forage donné.

        L'objectif est de maximiser la surface fragmentée tout en appliquant une 
        pénalité pour les zones en sur-fragmentation. Le score est calculé selon 
        la formule : Score = Surface_Frag - (λ * Surface_Over).

        Args:
            holes (np.ndarray): Liste des coordonnées des trous à évaluer.

        Returns:
            float: Le score final. Un score plus élevé indique un meilleur plan 
                   de forage pour l'optimiseur.
        """
        A_frag, A_frag_over = self.fragmentation_grid.evaluate_fragmentation_metrics(holes)
        return A_frag - (LAMDBA * A_frag_over)
    
    def filter_holes(self, holes: np.ndarray) -> np.ndarray:
        """
        Filtre les trous de forage selon les contraintes géométriques.

        Cette méthode applique deux filtres successifs :
        1. Filtre Spatial : Ne conserve que les trous situés à l'intérieur ou sur le bord 
           du polygone de forage.
        2. Filtre de Sécurité : Élimine les trous situés trop près des zones de danger (bootlegs).

        Args:
            holes (np.ndarray): Matrice brute des coordonnées des trous (n_holes, 2).

        Returns:
            np.ndarray: Une matrice filtrée ne contenant que les coordonnées des trous valides.
        """
        # Remove points outside of the polygon
        mask_in = vectorized.contains(self.polygon, holes[:, 0], holes[:, 1]) | vectorized.touches(self.polygon, holes[:, 0], holes[:, 1])
        holes = holes[mask_in]

        # Remove points too close of bootlegs
        mask_dist = self.fragmentation_grid.get_mask_dist_from_bootlegs(holes)
        holes = holes[mask_dist]
        return holes

    def determine_initial_solution(self):
        """
        Génère une première configuration de forage en maille quinconce.
        
        L'algorithme crée une grille régulière, puis filtre les points 
        selon les contraintes géométriques.
        """
        if len(self.current_holes) != 0:
            return # An initial solution exists

        holes = np.empty((0, 2))
        y_range = np.arange(self.fragmentation_grid.y_min - B, self.fragmentation_grid.y_max + B, B)
        for i, y in enumerate(y_range):
            offset = (S / 2) if (i % 2 == 0) else 0
                
            x_range = np.arange(self.fragmentation_grid.x_min - S, self.fragmentation_grid.x_max + S, S) + offset

            holes_row = np.column_stack((x_range, np.full_like(x_range, y)))
        
            holes = np.vstack((holes, holes_row))

        self.current_holes_non_filtered = holes

        holes = self.filter_holes(holes)

        self.current_objective = self.get_objective(holes)
        self.current_holes = holes

    def improve_solution_by_simulated_annealing(self):
        """
        Affine le plan de forage via une métaheuristique de Recuit Simulé.
        
        L'algorithme explore différents décalages (dx, dy) de la grille initiale. 
        Il accepte systématiquement les améliorations et accepte les dégradations 
        avec une probabilité décroissante (température) pour s'extraire des optimums locaux.
        
        Paramètres clés :
            - T_init / T_min : Température de départ et d'arrêt.
            - cooling_param : Vitesse de refroidissement.
            - history : Mémorisation pour éviter de recalculer des décalages déjà testés.
        """
        # SIMULATED ANNEALING PARAMS 
        T_init = 50
        T_min = 0.001
        cooling_param = 0.95   
        history= {}
        iteration = 0

        # INITIAL SOLUTION
        temp = T_init
        current_obj, current_holes = self.current_objective, self.current_holes
        curr_dx, curr_dy = self.current_dx, self.current_dy
        history[(curr_dx, curr_dy)] = (current_obj, current_holes)

        best_obj = current_obj
        best_holes = current_holes
        best_dx, best_dy = curr_dx, curr_dy
        
        # NEIGHBORHOUD PARAMS
        max_dx = S - RESOLUTION 
        max_dy = B - RESOLUTION
        max_i_step = int(max_dx / RESOLUTION)
        max_j_step = int(max_dy / RESOLUTION)

        with open("data/algo.log", "w", encoding="utf-8") as f:
            while temp > T_min:
                iteration += 1

                range_i = max(5, int(max_i_step * (temp / T_init)))
                range_j = max(5, int(max_j_step * (temp / T_init)))

                jump_i = np.random.randint(-range_i, range_i + 1)
                jump_j = np.random.randint(-range_j, range_j + 1)

                new_i = np.clip(round(curr_dx / RESOLUTION) + jump_i, 0, max_i_step)
                new_j = np.clip(round(curr_dy / RESOLUTION) + jump_j, 0, max_j_step)

                new_dx = round(new_i * RESOLUTION, 1)
                new_dy = round(new_j * RESOLUTION, 1)

                if new_dx == curr_dx and new_dy == curr_dy:
                    temp *= cooling_param
                    continue

                pos = (new_dx, new_dy)
                if pos in history:
                    new_obj, new_holes = history[pos]
                else:
                    new_holes = self.filter_holes(self.current_holes_non_filtered + [new_dx, new_dy])
                    new_obj = self.get_objective(new_holes)
                    history[pos] = (new_obj, new_holes)
                    
                delta = new_obj - current_obj

                if delta > 0 or np.random.rand() < np.exp(delta / temp):
                    curr_dx, curr_dy = new_dx, new_dy
                    current_obj = new_obj      
                    current_holes = new_holes

                    print(f'iteration: {iteration}, delta:{delta}, current obj: {current_obj}, curr_dx:{curr_dx}, curr_dy:{curr_dy}', file=f, flush=True)

                    if current_obj > best_obj:
                        best_obj = current_obj
                        best_holes = current_holes
                        best_dx, best_dy = curr_dx, curr_dy

                temp *= cooling_param

            if best_obj > self.current_objective:
                self.current_objective = best_obj
                self.current_holes = best_holes
                self.current_dx, self.current_dy = best_dx, best_dy
            self.save_holes_to_json()

    def save_holes_to_json(self, filename="data/output.json"):
        """
        Exporte la configuration finale des trous de forage vers un fichier JSON.

        Args:
            filename (str): Chemin du fichier de destination. Par défaut "data/output.json".
        """
        data = {
            "holes": self.current_holes.tolist() if hasattr(self.current_holes, 'tolist') else self.current_holes
        }

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def plot(self):
        """
        Génère une figure Matplotlib représentant le plan de forage final.

        Returns:
            matplotlib.figure.Figure: L'objet figure contenant le rendu visuel.
        """
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.grid(True, which='major', linestyle='-', linewidth='0.5', color='gray', alpha=0.5)

        # Polygon
        x_b, y_b = self.polygon.exterior.xy
        ax.plot(x_b, y_b, color='black', linewidth=1, label='Inner boundary', linestyle='dashed')
        ax.fill(x_b, y_b, color='lightgreen', alpha=0.3, zorder=0)

        # Bootlegs
        for bootleg in self.bootlegs:
            circle = Circle((bootleg[0], bootleg[1]), BOOTLEG_DIST, fill=False, linestyle='dashed', edgecolor='purple', alpha=0.3)
            ax.add_patch(circle)

        # Holes
        if len(self.current_holes) > 0:
            ax.scatter(self.current_holes[:, 0], self.current_holes[:, 1], c='green', edgecolors='black', s=30, label=f'Holes ({len(self.current_holes)})')
            
        ax.set_aspect('equal')
        ax.legend()

        A_total = self.polygon.area
        A_frag, A_frag_over = self.fragmentation_grid.evaluate_fragmentation_metrics(self.current_holes)
        frag_perc = A_frag / A_total * 100
        frag_over_perc = A_frag_over / A_total * 100
        ax.set_title('Final fragmentation Field \n'
                     f'Fragmented: {round(A_frag, 1)}m² / {round(A_total,1)}m² ({round(frag_perc)}%) | Over-frag (F>={T_OVER}): {round(A_frag_over, 1)}m² ({round(frag_over_perc)}%)| {len(self.current_holes)} holes')

        return fig