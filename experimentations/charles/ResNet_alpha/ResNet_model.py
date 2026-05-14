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

        # Paramètre apprenable alpha : nombre de véhicules cachés par gap
        init_alpha = torch.ones(n_gaps, dtype=torch.float32) * (self.N / n_gaps)
        self.alpha = nn.Parameter(init_alpha)

    def physics_activation(self, gap):
        """Loi de Greenshields : v(rho) = vmax * (1 - rho/rho_max)"""
        rho = self.alpha / (gap + 1e-6)
        v = self.v_max * (1.0 - rho / self.rho_max)
        return torch.clamp(v, min=0.0, max=self.v_max)

    def forward(self, return_history=False):
        """
        Dépliage temporel.
        :param return_history: Si True, retourne l'historique complet des positions pour la visualisation.
        """
        import numpy as np  # Import local pour le formattage history

        x = self.x_0_followers.clone()

        if return_history:
            history_followers = [x.detach().numpy().copy()]
            history_leader = [self.x_0_leader.item()]

        for step in range(self.num_steps):
            t = step * self.dt

            x_leader_t = self.x_0_leader + self.v_max * t
            x_next = torch.cat([x[1:], torch.tensor([x_leader_t], device=x.device)])

            gap = x_next - x
            v = self.physics_activation(gap)

            # Euler Explicit (Residual Connection)
            x = x + v * self.dt

            if return_history:
                history_followers.append(x.detach().numpy().copy())
                history_leader.append(x_leader_t.item())

        if return_history:
            return x, np.array(history_followers), np.array(history_leader)
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
