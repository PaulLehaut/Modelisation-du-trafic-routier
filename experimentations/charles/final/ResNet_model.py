import torch
import torch.nn as nn

# =====================================================================
# ARCHITECTURE RESNET POUR L'INTÉGRATION D'EDO (PINN)
# =====================================================================


class TrafficResNet(nn.Module):
    def __init__(
        self,
        n_gaps,
        N_total,
        L_v,
        v_max,
        rho_max,
        dt_h,
        num_steps,
        x_0_followers,
        x_0_leader,
        z_bar,
    ):
        super(TrafficResNet, self).__init__()
        self.N = N_total
        self.L_v = L_v
        self.v_max = v_max
        self.rho_max = rho_max
        self.dt = dt_h
        self.num_steps = num_steps

        self.x_0_followers = x_0_followers
        self.x_0_leader = x_0_leader
        self.z_bar = z_bar

        init_alpha = torch.ones(n_gaps, dtype=torch.float32) * (self.N / n_gaps)
        self.alpha = nn.Parameter(init_alpha)

    def physics_activation(self, gap):
        """Loi de Greenshields : v(rho) = vmax * (1 - rho/rho_max)"""
        rho = self.alpha / (gap + 1e-6)
        v = self.v_max * (1.0 - rho / self.rho_max)
        return torch.clamp(v, min=0.0, max=self.v_max)

    def forward(self, return_history=False, require_grad_history=False):
        """[MODIFICATION]

        :param require_grad_history: Si True, retourne l'historique complet en
        tant que
                                     tenseur PyTorch (conserve le graphe de
                                     calcul pour la loss continue).
        """
        import numpy as np

        x = self.x_0_followers.clone()

        # [MODIFICATION] Initialisation des listes d'historique adaptées
        if require_grad_history:
            history_tensor = [x]

        if return_history:
            history_followers = [x.detach().numpy().copy()]
            history_leader = [self.x_0_leader.item()]

        # [AJOUT - Suivi post-traitement] Stockage des densités intermédiaires au cours du temps
        intermediate_rhos = []

        for step in range(self.num_steps):
            t = step * self.dt
            x_leader_t = self.x_0_leader + self.v_max * t
            x_next = torch.cat([x[1:], torch.tensor([x_leader_t], device=x.device)])

            gap = x_next - x
            v = self.physics_activation(gap)

            # [AJOUT] Enregistrement de la densité intermédiaire pour analyse post-traitement
            with torch.no_grad():
                current_rho = self.alpha / (gap + 1e-6)
                intermediate_rhos.append(current_rho.detach().cpu().numpy())

            # Euler Explicit (Residual Connection)
            x = x + v * self.dt

            # [MODIFICATION] Stockage différenciable pour l'Approche 3
            if require_grad_history:
                history_tensor.append(x)

            if return_history:
                history_followers.append(x.detach().numpy().copy())
                history_leader.append(x_leader_t.item())

        # [MODIFICATION] Retour conditionnel selon le besoin (Entraînement vs Inférence)
        if require_grad_history:
            return torch.stack(history_tensor)  # Shape: (num_steps + 1, n_followers)

        if return_history:
            return (
                x,
                np.array(history_followers),
                np.array(history_leader),
                np.array(intermediate_rhos),
            )  # [MODIFICATION] Ajout des densités intermédiaires au retour

        return x


def project_alpha(alpha, z_bar, N_total):
    """Projection dans l'Espace Admissible A_N (Eq 17)"""
    with torch.no_grad():
        alpha.clamp_(min=1.0)
        alpha = torch.min(alpha, z_bar)
        sum_alpha = torch.sum(alpha)
        alpha.mul_(N_total / sum_alpha)
        alpha.clamp_(min=1.0)
    return alpha
