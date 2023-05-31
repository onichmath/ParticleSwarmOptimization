import numpy as np 
from matplotlib import pyplot as plt
from matplotlib import animation, cm
from time import perf_counter

class Particle():
    """Represents a particle"""

    @classmethod
    def decrement_weight(cls):
        """Decrement the weight/inertia of all particles"""
        if cls.weight > 0.3:
            cls.weight = cls.weight - 0.01

    @classmethod
    def setup(cls, obj_func, weight:float=1.0, social:float=1.5, cognitive:float=1.5, dimensions:int=2, upper:float=5.0, lower:float=-5.0, target:list=[0,0], error:float=1e-6) -> None:
        """Setup the swarm space"""
        cls.weight = weight
        cls.social_coefficient = social
        cls.cognitive_coefficient = cognitive
        cls.dimensions = dimensions
        cls.gbest_pos = [upper for d in range(dimensions)]
        cls.upper_bound = upper
        cls.lower_bound = lower
        cls.target = target
        cls.error = error
        cls.fitness = obj_func
        assert len(cls.gbest_pos) == cls.dimensions

    @classmethod
    def within_target_error(cls) -> bool:
        """Returns true if gbest_pos is within the designated target error"""
        if np.abs(cls.gbest_pos[0]) < np.abs(cls.target[0] + cls.error):
            return True
        return False

        #     if cls.gbest_pos[0] > cls.target[0] - cls.error:
        #         if cls.gbest_pos[1] < cls.target[1] + cls.error:
        #             if cls.gbest_pos[1] > cls.target[1] - cls.error:
        #                 return True
        # return False 
        #
    @classmethod
    def dist_target(cls):
        """Returns the distance from the target in each dimension"""
        dist = [None,None]

    def __init__(self) -> None:
        # Set position. Makes vector with dimensions Particle.dimensions
        self.position = np.random.uniform(low=Particle.lower_bound, high=Particle.upper_bound, size=Particle.dimensions)
        # Set best known pos
        self.pbest_pos = self.position
        self.update_gbest_pos()
        # Initialize velocity within bounds.
        self.velocity = np.random.uniform(low=Particle.lower_bound, high=Particle.upper_bound, size=Particle.dimensions)

    def __str__(self) -> str:
        """Returns the position and velocity of the particle"""
        return f"{hex(id(self))} current position is: {self.position}. My pbest is: {self.pbest_pos}. My velocity is: {self.velocity}."

    def update_position(self) -> None:
        """Set the particle's position based on current position and velocity"""
        self.position = self.position + self.velocity

    def update_pbest_pos(self) -> bool:
        """Update particle's best known pos with its current position, if its better. Returns True/false if updated/not."""
        if self.fitness(self.position) < self.fitness(self.pbest_pos):
            self.pbest_pos = self.position
            return True
        return False

    def update_gbest_pos(self) -> None:
        """Updates gbest based on particle index"""
        if self.fitness(self.pbest_pos) < self.fitness(Particle.gbest_pos):
            Particle.gbest_pos = self.pbest_pos 

    def update_velocity(self) -> None: 
        """Update the particle's velocity in each dimension"""
        for d,pos in enumerate(self.position):
            # Inertia * velocity
            inertial_velocity =  Particle.weight * self.velocity[d]
            # Find distance to personal best pos 
            dist_pbest = self.pbest_pos[d] - self.position[d]
            # Find distance to global best pos
            dist_gbest = Particle.gbest_pos[d] - self.position[d]
            # Set cognitive constant
            p_const = Particle.cognitive_coefficient * np.random.uniform(low=0, high=1)
            # Set the social constant
            g_const = Particle.social_coefficient * np.random.uniform(low=0, high=1)
            # Set velocity in given dimension
            final_velocity = inertial_velocity + p_const * dist_pbest + g_const * dist_gbest 
            self.velocity[d] = final_velocity

    def enforce_bounds(self) -> None:
        """When the position is outside of bounds, it is set in bounds. When a velocity is outside, it is set to 0"""
        for d,pos in enumerate(self.position):
            if self.position[d] > Particle.upper_bound:
                self.position[d] = Particle.upper_bound - (self.position[d] / 1000)
            elif self.position[d] < Particle.lower_bound:
                self.position[d] = Particle.lower_bound - (self.position[d] / 1000)
            if self.velocity[d] > Particle.upper_bound:
                self.velocity[d] = -0.1
            elif self.velocity[d] < Particle.lower_bound:
                self.velocity[d] = 0.1

    def search(self) -> None:
        """A single particle's search for minimum"""
        self.update_velocity()
        self.update_position()
        self.enforce_bounds()
        if self.update_pbest_pos():
            self.update_gbest_pos()
            
# -----------------------------------------------------
def setup_plot(type3d:bool=True):
    """Based off of https://machinelearningmastery.com/a-gentle-introduction-to-particle-swarm-optimization/
    and https://towardsdatascience.com/swarm-intelligence-coding-and-visualising-particle-swarm-optimisation-in-python-253e1bd00772"""
    l = Particle.lower_bound
    u = Particle.upper_bound
    x, y = np.array(np.meshgrid(np.linspace(l,u,100), np.linspace(l,u,100)))
    z = Particle.fitness([x, y])
    x_min = x.ravel()[z.argmin()]
    y_min = y.ravel()[z.argmin()]
    fig =plt.figure(figsize=(8,8))
    if type3d:
        ax = fig.add_subplot(111, projection="3d")
        ax.plot([x_min], [y_min], marker='x', markersize=15, color="red")
        ax.plot_wireframe(x,y,z, color='red', rcount=500, ccount=500, linewidth=0.4, alpha=0.4)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        return fig, ax
    else:
        plt.imshow(z, extent=[l, u, l, u], origin='lower', cmap='viridis', alpha=0.5)
        plt.colorbar()
        contours = plt.contour(x, y, z, 10, colors='black', alpha=0.4)
        plt.clabel(contours, inline=True, fontsize=8, fmt="%.0f")
        return fig

def visualize_function(obj_func, l, u) -> None:
    """Visualize an objective function. Takes an objective function meant to be used as classmethod as parameter, so first parameter for it is set to None"""
    x, y = np.array(np.meshgrid(np.linspace(l,u,100), np.linspace(l,u,100)))
    z = obj_func(None, [x, y])
    x_min = x.ravel()[z.argmin()]
    y_min = y.ravel()[z.argmin()]
    fig =plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(x, y, z, rcount=50, ccount=50, cmap=cm.Blues)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()

# -------------------------------------------------------------
def particle_swarm_optimization(obj_func, social:float=1.5, cognitive:float=1.5, weight:float=1.0, upper:float=5.0, lower:float=-5.0, dec_weight:int=True, n_particles:int=5, iterations:int=50, type3d:bool=True, dimensions:int=2, target:list=[0,0], error:float=1e-6):
    # Swarm Setup
    Particle.setup(obj_func=obj_func, social=social, cognitive=cognitive, weight=weight, dimensions=dimensions, upper=upper, lower=lower, target=target, error=error)
    particles = [Particle() for n in range(n_particles)]
    # Matplotlib setup call
    if type3d:
        fig, ax = setup_plot(type3d=True)
    else:
        fig = setup_plot(type3d=False)
    artists = []

    i = 0
    start = perf_counter()
    while i < iterations:
        for particle in particles:
            particle.search()
        if dec_weight == True:
            Particle.decrement_weight
        i += 1

        # Matplotlib frames for animation
        x_positions = [particles[i].position[0] for i,part in enumerate(particles)]
        y_positions = [particles[i].position[1] for i,part in enumerate(particles)]
        if type3d:
            fitness_vals = [Particle.fitness([particles[i].position[0], particles[i].position[1]]) for i,part in enumerate(particles)]
            frame = ax.scatter(xs=x_positions,ys=y_positions, zs=fitness_vals, c='b', marker='$P$')
            title = plt.figtext(x=0, y=0.9, s=f"Iteration {i}\nGbest {Particle.gbest_pos}\nTarget {Particle.target} Error {Particle.error}\n{(perf_counter() - start) * 1000}ms")
        else:
            frame = plt.scatter(x_positions, y_positions, c='b', marker='$P$')
            title = plt.text(x=-4, y=5.5, s=f"PSO Iteration {i}, Current Gbest is {Particle.gbest_pos}, {(perf_counter() - start) * 1000} Milliseconds")
        artists.append([frame, title])

        if Particle.within_target_error():
            break
    end = perf_counter()

    print(f"Gbestpos is: {Particle.gbest_pos}. {i} iterations. {(end - start) * 1000} Milliseconds")
    anim = animation.ArtistAnimation(fig=fig, artists=artists, repeat=False, interval=100)
    plt.show()


# ----------------------------------------------------------
def Ackleys(cls, position):
    """Ackley's function from wikipedia."""
    x = position[0]
    y = position[1]
    return -20.0 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2))) - np.exp(0.5 * (np.cos(2 * 
      np.pi * x)+np.cos(2 * np.pi * y))) + np.e + 20

def paraboloid(cls, position):
    """Ackley's function from wikipedia"""
    x = position[0]
    y = position[1]
    return x**2 + y**2

# --------------------------------------------------------------

def main():
    particle_swarm_optimization(obj_func=classmethod(paraboloid), type3d=True, social=1.5, cognitive=1.5, weight=0.2, n_particles=500, dec_weight=False, iterations=500)
    # particle_swarm_optimization(obj_func=classmethod(Ackleys), type3d=True, social=1.5, cognitive=1.5, weight=0.8, n_particles=500, dec_weight=True, iterations=500)
    # visualize_function(paraboloid, l=-5.0, u=5.0)

if __name__ == "__main__":
    main()
