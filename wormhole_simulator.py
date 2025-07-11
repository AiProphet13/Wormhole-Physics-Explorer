import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider, Button, RadioButtons
from scipy.integrate import solve_ivp
from scipy.constants import G, c
import matplotlib.animation as animation
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.interpolate import griddata

class WormholeSimulator:
    def __init__(self, throat_radius=1.0, shape_exponent=1.0, exotic_density=0.1, 
                 impact_parameter=1.2, mass_ratio=0.001, redshift_factor=0.0):
        """Initialize the wormhole simulator with advanced physics."""
        self.throat_radius = throat_radius
        self.shape_exponent = shape_exponent
        self.exotic_density = exotic_density
        self.impact_parameter = impact_parameter
        self.mass_ratio = mass_ratio
        self.redshift_factor = redshift_factor
        self.n_points = 200
        self.l_max = 10.0
        self.view_mode = "Embedding Diagram"
        self.fig = plt.figure(figsize=(20, 16), facecolor='black')
        self.setup_ui()
        self.update_wormhole()
        self.fig.patch.set_facecolor('black')
        for ax in [self.ax, self.density_ax, self.gw_ax, self.tidal_ax, self.time_ax]:
            ax.set_facecolor('black')
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.title.set_color('cyan')

    def shape_function(self, l):
        """Morris-Thorne shape function with adjustable exponent."""
        return np.sqrt(self.throat_radius**2 + (l / self.shape_exponent)**2)

    def redshift_function(self, l):
        """Redshift function for time dilation effects."""
        return np.exp(-self.redshift_factor * (l**2) / (1 + l**2 + 1e-10))  # Avoid div by zero

    def shape_derivative(self, l):
        """First derivative of shape function."""
        epsilon = 1e-10
        denom = np.sqrt(1 + (l / (self.shape_exponent * self.throat_radius + epsilon))**2)
        return (l / self.shape_exponent**2) / denom

    def shape_second_derivative(self, l):
        """Second derivative of shape function for tidal forces."""
        epsilon = 1e-10
        term1 = 1 / (self.shape_exponent**2 * np.sqrt(1 + (l / (self.shape_exponent * self.throat_radius + epsilon))**2))
        term2 = (l**2 / (self.shape_exponent**4 * (self.throat_radius + epsilon)**2)) / (1 + (l / (self.shape_exponent * self.throat_radius + epsilon))**2)**(3/2)
        return term1 - term2

    def energy_density(self, l):
        """Exotic matter energy density (negative for stability)."""
        return -self.exotic_density * np.exp(-(l / (self.throat_radius * 2))**2)

    def tidal_force(self, l, object_size=1.0):
        """Tidal acceleration at position l (m/s²)."""
        epsilon = 1e-10
        b = self.shape_function(l)
        db_dr = self.shape_derivative(l)
        d2b_dr2 = self.shape_second_derivative(l)
        radial_term = -c**2 * (d2b_dr2 - db_dr / (np.abs(l) + epsilon)) * object_size / b
        lateral_term = c**2 * (db_dr / (np.abs(l) + epsilon)) * object_size / b
        return {
            'radial': radial_term,
            'lateral': lateral_term,
            'total': np.sqrt(radial_term**2 + lateral_term**2)
        }

    def time_dilation_factor(self, l):
        """Time dilation factor relative to infinity, handling l=0."""
        epsilon = 1e-10
        b_over_l = np.where(np.abs(l) > epsilon, self.shape_function(l) / l, 0)  # Limit as l→0 is 0
        return np.sqrt(1 - b_over_l**2) * self.redshift_function(l)

    def _calculate_data(self):
        """Compute all data for visualization."""
        radial_coords = np.linspace(-self.l_max, self.l_max, self.n_points)
        theta_angles = np.linspace(0, 2 * np.pi, self.n_points)
        L_grid, Theta_grid = np.meshgrid(radial_coords, theta_angles)
        R = self.shape_function(L_grid)
        X = R * np.cos(Theta_grid)
        Y = R * np.sin(Theta_grid)
        Z = L_grid
        energy = self.energy_density(radial_coords)
        self.calculate_geodesics()
        energy_conditions = self.check_energy_conditions()
        gw_f, gw_h = self.calculate_gravitational_waves()
        entanglement = self.create_entanglement_bridge()
        tidal_forces = {k: np.array([self.tidal_force(l_val)[k] for l_val in radial_coords]) for k in ['radial', 'lateral', 'total']}
        time_dilation = np.array([self.time_dilation_factor(l_val) for l_val in radial_coords])
        if self.view_mode == "Observer View":
            self.calculate_observer_view()
        return {
            'X': X, 'Y': Y, 'Z': Z, 'radial_coords': radial_coords,
            'energy': energy, 'energy_conditions': energy_conditions,
            'gw_f': gw_f, 'gw_h': gw_h, 'entanglement': entanglement,
            'tidal_forces': tidal_forces, 'time_dilation': time_dilation
        }

    def calculate_geodesics(self):
        """Compute light paths with deflection or capture."""
        self.geodesics = []
        num_rays = 24 if self.view_mode == "Observer View" else 8
        angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)
        for angle in angles:
            def dz_dphi(phi, z):
                r = self.shape_function(z[0])
                arg = 1 - self.impact_parameter**2 / r**2
                return [-(r**2 / self.impact_parameter) * np.sqrt(max(arg, 0))]

            def turning_event(phi, z):
                r = self.shape_function(z[0])
                return 1 - self.impact_parameter**2 / r**2
            turning_event.terminal = True
            turning_event.direction = -1

            try:
                sol_in = solve_ivp(dz_dphi, [0, np.pi], [self.l_max], events=turning_event, dense_output=True, max_step=0.1, atol=1e-8, rtol=1e-8)
                if sol_in.t_events[0].size > 0:
                    phi_in = np.linspace(0, sol_in.t_events[0][0], 100)
                    z_in = sol_in.sol(phi_in)[0]
                    sol_out = solve_ivp(lambda phi, z: -dz_dphi(phi, z), [sol_in.t_events[0][0], sol_in.t_events[0][0] + np.pi], [sol_in.y_events[0][0][0]], dense_output=True, max_step=0.1, atol=1e-8, rtol=1e-8)
                    phi_out = np.linspace(sol_in.t_events[0][0], sol_in.t_events[0][0] + np.pi, 100)
                    z_out = sol_out.sol(phi_out)[0]
                    phi = np.concatenate((phi_in, phi_out))
                    z = np.concatenate((z_in, z_out))
                else:
                    phi = np.linspace(0, np.pi, 100)
                    z = sol_in.sol(phi)[0]
                r_vals = self.shape_function(z)
                x = r_vals * np.cos(phi + angle)
                y = r_vals * np.sin(phi + angle)
                self.geodesics.append((x, y, z, phi, r_vals))
            except Exception:
                # Silent fallback
                z = np.linspace(-self.l_max, self.l_max, 100)
                r_vals = self.shape_function(z)
                x = r_vals * np.cos(angle) * np.ones_like(z)
                y = r_vals * np.sin(angle) * np.ones_like(z)
                self.geodesics.append((x, y, z, np.zeros_like(z), r_vals))

    def calculate_observer_view(self):
        """Generate background star field for observer view."""
        num_stars = 500
        star_angles = np.random.uniform(0, 2*np.pi, num_stars)
        star_distances = np.random.uniform(10, 100, num_stars)
        self.star_field = []
        for angle, dist in zip(star_angles, star_distances):
            x = dist * np.cos(angle)
            y = dist * np.sin(angle)
            z = 0
            self.star_field.append(np.array([x, y, z]))

    def render_observer_view(self):
        """Optimized ray-tracing for observer view with vectorization."""
        if not hasattr(self, 'star_field') or not self.geodesics:
            return np.zeros((100, 100, 3))  # Fallback blank image

        observer_pos = np.array([0, 0, self.l_max])
        fov = np.pi / 3
        img_size = 200  # Reduced for speed; increase for quality
        image = np.zeros((img_size, img_size, 3))

        # Grid of viewing angles
        xv = np.linspace(-fov/2, fov/2, img_size)
        yv = np.linspace(-fov/2, fov/2, img_size)
        xx, yy = np.meshgrid(xv, yv)
        dir_vecs = np.stack([np.sin(xx), np.sin(yy), -np.cos(xx)], axis=-1)
        dir_vecs /= np.linalg.norm(dir_vecs, axis=-1)[..., np.newaxis]

        # Precompute ray_vecs for all geodesics
        all_ray_vecs = [np.array(ray[:3]).T for ray in self.geodesics]

        # Vectorized min_dist calculation across all rays
        min_dists = np.full((img_size, img_size), np.inf)
        min_indices = np.full((img_size, img_size), -1)
        min_ray_ids = np.full((img_size, img_size), -1)
        for ray_id, ray_vec in enumerate(all_ray_vecs):
            diffs = ray_vec[np.newaxis, np.newaxis, ...] - observer_pos
            dists = np.linalg.norm(diffs, axis=-1)
            mask = dists.min(axis=-1) < min_dists
            min_dists[mask] = dists.min(axis=-1)[mask]
            min_indices[mask] = dists.argmin(axis=-1)[mask]
            min_ray_ids[mask] = ray_id

        # Wormhole hits
        hit_mask = min_dists < 0.5 * self.throat_radius
        z_vals = np.full((img_size, img_size), 0.0)
        for i in range(img_size):
            for j in range(img_size):
                if hit_mask[i, j]:
                    ray_id = min_ray_ids[i, j]
                    min_idx = min_indices[i, j]
                    z_vals[i, j] = self.geodesics[ray_id][2][min_idx]

        colors = np.where(z_vals[..., np.newaxis] < 0, [0.8, 0.2, 0.2], [0.2, 0.2, 0.8])
        image[hit_mask] = colors[hit_mask]

        # Einstein ring
        r = np.sqrt(xx**2 + yy**2)
        ring_mask = np.abs(r - 0.2 * self.throat_radius) < 0.05 * self.throat_radius
        image[ring_mask] = [1.0, 1.0, 0.8]

        # Background stars (vectorized)
        star_dirs = np.array(self.star_field) - observer_pos
        star_dirs /= np.linalg.norm(star_dirs, axis=-1)[:, np.newaxis]
        dot_products = np.dot(star_dirs, dir_vecs.reshape(-1, 3).T).T.reshape(img_size, img_size, -1)
        star_mask = np.any(dot_products > 0.999, axis=-1)
        image[star_mask] = np.maximum(image[star_mask], [1.0, 1.0, 1.0] * 0.8)

        return image

    def check_energy_conditions(self):
        """Evaluate energy conditions and traversability."""
        T_00 = self.energy_density(0)
        NEC_violation = T_00 < 0
        radial_coords = np.linspace(-self.l_max, self.l_max, 100)
        ANEC_integral = np.trapz(self.energy_density(radial_coords), radial_coords)
        tidal_throat = self.tidal_force(0)['total']
        time_dilation_throat = self.time_dilation_factor(0)
        return {
            'NEC_violation': NEC_violation,
            'ANEC_integral': ANEC_integral,
            'is_traversable': ANEC_integral < 0 and abs(tidal_throat) < 1e16,
            'tidal_throat': tidal_throat,
            'time_dilation_throat': time_dilation_throat
        }

    def calculate_gravitational_waves(self):
        """Gravitational wave spectrum from traversing object."""
        f_peak = c / (2 * np.pi * self.throat_radius)
        h_c = (G * self.mass_ratio / c**2) / self.throat_radius
        f = np.logspace(np.log10(f_peak) - 2, np.log10(f_peak) + 2, 100)
        h = h_c * (f / f_peak)**(-1/3) * np.exp(-(f / f_peak)**4)
        return f, h

    def create_entanglement_bridge(self):
        """Quantum entanglement metrics via ER=EPR."""
        fidelity = np.exp(-self.l_max / (self.throat_radius * 10))
        decoherence_time = (self.throat_radius**2 * c**3) / (G * self.exotic_density)
        return {
            'fidelity': fidelity,
            'decoherence_time': decoherence_time,
            'quantum_bandwidth': c / (2 * self.throat_radius)
        }

    def setup_ui(self):
        """Set up plots and interactive controls with cosmic theme."""
        self.ax = self.fig.add_subplot(231, projection='3d')
        self.ax.set_position([0.02, 0.35, 0.4, 0.6])
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title('Wormhole Visualization', color='cyan')

        self.density_ax = self.fig.add_subplot(234)
        self.density_ax.set_position([0.02, 0.1, 0.4, 0.2])
        self.density_ax.set_title('Exotic Matter Distribution', color='cyan')
        self.density_ax.set_xlabel('Radial Coordinate (l)')
        self.density_ax.set_ylabel('Energy Density')
        self.density_ax.grid(True, color='gray', alpha=0.3)

        self.gw_ax = self.fig.add_subplot(232)
        self.gw_ax.set_position([0.45, 0.6, 0.25, 0.3])
        self.gw_ax.set_title('Gravitational Wave Spectrum', color='cyan')
        self.gw_ax.set_xlabel('Frequency (Hz)')
        self.gw_ax.set_ylabel('Characteristic Strain')
        self.gw_ax.grid(True, color='gray', alpha=0.3)
        self.gw_ax.set_xscale('log')
        self.gw_ax.set_yscale('log')

        self.tidal_ax = self.fig.add_subplot(233)
        self.tidal_ax.set_position([0.72, 0.6, 0.25, 0.3])
        self.tidal_ax.set_title('Tidal Forces', color='cyan')
        self.tidal_ax.set_xlabel('Radial Coordinate (l)')
        self.tidal_ax.set_ylabel('Tidal Acceleration (m/s²)')
        self.tidal_ax.grid(True, color='gray', alpha=0.3)
        self.tidal_ax.set_yscale('log')

        self.time_ax = self.fig.add_subplot(236)
        self.time_ax.set_position([0.45, 0.1, 0.25, 0.2])
        self.time_ax.set_title('Time Dilation', color='cyan')
        self.time_ax.set_xlabel('Radial Coordinate (l)')
        self.time_ax.set_ylabel('Time Dilation Factor')
        self.time_ax.grid(True, color='gray', alpha=0.3)

        self.diag_ax = self.fig.add_subplot(235)
        self.diag_ax.set_position([0.72, 0.1, 0.25, 0.4])
        self.diag_ax.axis('off')
        self.diag_ax.set_title('Wormhole Diagnostics', fontsize=14, color='cyan')

        slider_y = 0.05
        slider_width = 0.3
        slider_height = 0.03

        ax_throat = plt.axes([0.05, slider_y, slider_width, slider_height], facecolor='#222222')
        self.throat_slider = Slider(ax_throat, 'Throat Radius', 0.1, 3.0, valinit=self.throat_radius, color='#4444ff')

        ax_exponent = plt.axes([0.05, slider_y - 0.05, slider_width, slider_height], facecolor='#222222')
        self.exponent_slider = Slider(ax_exponent, 'Shape Exponent', 0.5, 3.0, valinit=self.shape_exponent, color='#4444ff')

        ax_density = plt.axes([0.05, slider_y - 0.10, slider_width, slider_height], facecolor='#222222')
        self.density_slider = Slider(ax_density, 'Exotic Matter Density', 0.01, 0.5, valinit=self.exotic_density, color='#ff4444')

        ax_impact = plt.axes([0.4, slider_y, slider_width, slider_height], facecolor='#222222')
        self.impact_slider = Slider(ax_impact, 'Impact Parameter', 0.1, 3.0, valinit=self.impact_parameter, color='#44ff44')

        ax_mass = plt.axes([0.4, slider_y - 0.05, slider_width, slider_height], facecolor='#222222')
        self.mass_slider = Slider(ax_mass, 'Object Mass Ratio', 1e-6, 1e-1, valinit=self.mass_ratio, valfmt='%1.0e', color='#ff8844')

        ax_redshift = plt.axes([0.4, slider_y - 0.10, slider_width, slider_height], facecolor='#222222')
        self.redshift_slider = Slider(ax_redshift, 'Redshift Factor', 0.0, 1.0, valinit=self.redshift_factor, color='#ff44ff')

        self.throat_slider.on_changed(self.update_parameter)
        self.exponent_slider.on_changed(self.update_parameter)
        self.density_slider.on_changed(self.update_parameter)
        self.impact_slider.on_changed(self.update_parameter)
        self.mass_slider.on_changed(self.update_parameter)
        self.redshift_slider.on_changed(self.update_parameter)

        reset_ax = plt.axes([0.75, 0.02, 0.1, 0.04], facecolor='#333333')
        self.reset_button = Button(reset_ax, 'Reset', color='#555555')
        self.reset_button.on_clicked(self.reset_parameters)

        animate_ax = plt.axes([0.6, 0.02, 0.1, 0.04], facecolor='#333333')
        self.animate_button = Button(animate_ax, 'Animate', color='#555555')
        self.animate_button.on_clicked(self.toggle_animation)

        view_ax = plt.axes([0.05, 0.9, 0.15, 0.05], facecolor='#333333')
        self.view_radio = RadioButtons(view_ax, ('Embedding Diagram', 'Observer View'), active=0)
        self.view_radio.on_clicked(self.change_view_mode)

        self.anim = None
        self.animating = False

        for slider in [self.throat_slider, self.exponent_slider, self.density_slider, self.impact_slider, self.mass_slider, self.redshift_slider]:
            slider.label.set_color('white')
            slider.valtext.set_color('white')

    def change_view_mode(self, label):
        """Switch between visualization modes."""
        self.view_mode = label
        self.update_wormhole()

    def toggle_animation(self, event):
        """Toggle smooth rotation animation using FuncAnimation."""
        self.animating = not self.animating
        if self.animating:
            self.animate_button.color = '#88ff88'
            self.anim = animation.FuncAnimation(self.fig, self._animate_frame, frames=np.arange(0, 360, 2), interval=20, blit=False)
        else:
            self.animate_button.color = '#555555'
            if self.anim:
                self.anim.event_source.stop()
        plt.draw()

    def _animate_frame(self, angle):
        """Animation frame update."""
        self.ax.view_init(elev=30, azim=angle)
        return [self.ax]

    def update_wormhole(self):
        """Update visualization with current parameters, avoiding recursion."""
        data = self._calculate_data()
        self.ax.clear()
        self.density_ax.clear()
        self.gw_ax.clear()
        self.tidal_ax.clear()
        self.time_ax.clear()
        self.diag_ax.cla()

        for ax in [self.ax, self.density_ax, self.gw_ax, self.tidal_ax, self.time_ax]:
            ax.set_facecolor('black')
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.title.set_color('cyan')
            ax.grid(True, color='gray', alpha=0.3)

        if self.view_mode == "Embedding Diagram":
            self.ax.plot_surface(data['X'], data['Y'], data['Z'], cmap='viridis', alpha=0.8, edgecolor='none')
            for angle in np.linspace(0, 2 * np.pi, 8, endpoint=False):
                r_line = self.shape_function(data['radial_coords'])
                x_line = r_line * np.cos(angle)
                y_line = r_line * np.sin(angle)
                z_line = data['radial_coords']
                self.ax.plot(x_line, y_line, z_line, 'w-', alpha=0.3, label='Structural Lines' if angle == 0 else None)
            for i, (x, y, z, phi, r_vals) in enumerate(self.geodesics):
                points = np.array([x, y, z]).T.reshape(-1, 1, 3)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                lc = Line3DCollection(segments, cmap='coolwarm', norm=plt.Normalize(0, self.l_max))
                lc.set_array(np.abs(z))
                self.ax.add_collection3d(lc)
                if i == 0:
                    self.ax.plot([], [], color='blue', label='Geodesic Paths (Color: |z|)')
            self.ax.legend(loc='upper right')
            num_stars = 100
            star_x = np.random.uniform(-50, 50, num_stars)
            star_y = np.random.uniform(-50, 50, num_stars)
            star_z = np.random.uniform(-50, 50, num_stars)
            star_sizes = np.random.uniform(10, 100, num_stars)
            self.ax.scatter(star_x, star_y, star_z, 'w*', s=star_sizes, alpha=0.7)
        else:  # Observer View
            self.observer_image = self.render_observer_view()
            self.ax.imshow(self.observer_image, extent=[-1, 1, -1, 1])
            self.ax.set_title("Observer View: What You Would See", color='cyan')
            self.ax.set_xlabel('Viewing Angle X')
            self.ax.set_ylabel('Viewing Angle Y')
            self.ax.grid(False)

        self.density_ax.plot(data['radial_coords'], data['energy'], 'm-', linewidth=2)
        self.density_ax.fill_between(data['radial_coords'], data['energy'], 0, color='magenta', alpha=0.3)
        self.density_ax.set_ylim(-0.6, 0.1)

        self.gw_ax.plot(data['gw_f'], data['gw_h'], 'y-', linewidth=2)
        self.gw_ax.set_title(f'GW from Object (Mass Ratio: {self.mass_ratio:.1e})')

        self.tidal_ax.plot(data['radial_coords'], np.abs(data['tidal_forces']['radial']), 'r-', label='Radial')
        self.tidal_ax.plot(data['radial_coords'], np.abs(data['tidal_forces']['lateral']), 'g-', label='Lateral')
        self.tidal_ax.plot(data['radial_coords'], np.abs(data['tidal_forces']['total']), 'b--', label='Total', linewidth=2)
        self.tidal_ax.legend()

        self.time_ax.plot(data['radial_coords'], data['time_dilation'], 'c-', linewidth=2)
        self.time_ax.set_ylim(0, 1.1)

        ec = data['energy_conditions']
        ent = data['entanglement']
        diagnostics_text = (
            "ENERGY CONDITIONS:\n"
            f"  NEC at Throat: {'VIOLATED' if ec['NEC_violation'] else 'SATISFIED'}\n"
            f"  ANEC Integral: {ec['ANEC_integral']:.3f}\n"
            f"  Traversable: {'YES' if ec['is_traversable'] else 'NO'}\n"
            f"  Tidal Force at Throat: {abs(ec['tidal_throat']):.3e} m/s²\n"
            f"  Time Dilation at Throat: {ec['time_dilation_throat']:.3f}\n\n"
            "QUANTUM ENTANGLEMENT:\n"
            f"  Fidelity: {ent['fidelity']:.3f}\n"
            f"  Decoherence Time: {ent['decoherence_time']:.3e} s\n"
            f"  Quantum Bandwidth: {ent['quantum_bandwidth']:.3e} Hz\n\n"
            "GEODESIC PARAMETERS:\n"
            f"  Impact Parameter: {self.impact_parameter:.2f}\n"
            f"  Redshift Factor: {self.redshift_factor:.2f}\n"
            f"  View Mode: {self.view_mode}"
        )
        self.diag_ax.text(0.02, 0.98, diagnostics_text, fontsize=10, color='white', va='top', ha='left', bbox=dict(facecolor='#111122', alpha=0.8))
        self.diag_ax.axis('off')

        self.ax.set_title(f'Morris-Thorne Wormhole (R={self.throat_radius:.2f})')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.view_init(elev=30, azim=45)
        plt.draw()

# Create and run the ultimate wormhole simulator
sim = WormholeSimulator(throat_radius=1.5, shape_exponent=1.5, exotic_density=0.2, impact_parameter=1.2, mass_ratio=0.001, redshift_factor=0.3)
plt.tight_layout()
plt.show()
