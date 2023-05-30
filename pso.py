import numpy as np 
from matplotlib import pyplot as plt
from matplotlib import animation

class Particle():
    """Represents a particle"""
    dimensions = 2
    # Inertial weight
    weight = 1 
    # Social and cognitive coefficients.
    # Cognitive makes a particle care more about its own findings
    # Social makes a particle care more about the swarm's findings
    social_coefficient = 1.5 
    cognitive_coefficient = 1.5
    # Upper and lower bounds of the problem
    upper_bound = 5
    lower_bound = -5
    # Swarm's best known position
    gbest_pos = [upper_bound] * dimensions
    # Target minimum values
    target = [0,0]
    error = 1e-6

    @classmethod
    def decrement_weight(cls):
        """Decrement the weight/inertia of all particles"""
        if cls.weight > 0.3:
            cls.weight = cls.weight - 0.01

    @classmethod
    def setup(cls, weight:float=1.0, social:float=1.5, cognitive:float=1.5, dimensions:int=2, upper:float=5.0, lower:float=-5.0, target:list=[0,0], error:float=1e-6) -> None:
        """Setup the swarm space"""
        cls.weight = weight
        cls.social_coefficient = social
        cls.cognitive_coefficient = cognitive
        cls.dimensions = dimensions
        cls.gbest_pos = [upper] * dimensions
        cls.upper_bound = upper
        cls.lower_bound = lower
        cls.target = target
        cls.error = error
        assert len(cls.gbest_pos) == cls.dimensions

    @classmethod
    def within_target_error(cls) -> bool:
        """Returns true if gbest_pos is within the designated target error"""
        for d in range(cls.dimensions):
            if cls.gbest_pos[d] < cls.target[d] + cls.error:
                if cls.gbest_pos[d] > cls.target[d] - cls.error:
                    return True 
        return False 

    @classmethod
    def fitness(cls, position):
        """Ackley's function from wikipedia"""
        x = position[0]
        y = position[1]
        # return x**2 + y**2
        return -20.0 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2))) - np.exp(0.5 * (np.cos(2 * 
          np.pi * x)+np.cos(2 * np.pi * y))) + np.e + 20

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
        return f"My current position is: {self.position}. My pbest is: {self.pbest_pos}. My velocity is: {self.velocity}."

    def update_position(self) -> None:
        """Set the particle's position based on current position and velocity"""
        self.position = self.position + self.velocity

    def update_pbest_pos(self) -> bool:
        """Update particle's best known pos with its current position, if its better. Returns True/false if updated/not."""
        if self.fitness(self.position) < self.fitness(self.pbest_pos):
            self.pbest_pos = self.position
            return True
        return False

    def update_gbest_pos(self) -> bool:
        """Updates gbest based on particle index"""
        if self.fitness(self.pbest_pos) < self.fitness(Particle.gbest_pos):
            Particle.gbest_pos = self.pbest_pos 
            return True
        return False

    def update_velocity(self) -> None: 
        """Update the particle's velocity in each dimension"""
        for d in range(len(self.velocity)):
            # Inertia * velocity
            inertial_velocity =  Particle.weight * self.velocity[d]
            # Find distance to personal best pos 
            dist_pbest = self.pbest_pos[d] - self.position[d]
            # Find distance to global best pos
            dist_gbest = Particle.gbest_pos[d] - self.position[d]
            # Set cognitive constant
            p_const = Particle.cognitive_coefficient * np.random.uniform(low=0.001, high=1)
            # Set the social constant
            g_const = Particle.social_coefficient * np.random.uniform(low=0.001, high=1)
            # Set velocity in given dimension
            final_velocity = inertial_velocity + p_const * dist_pbest + g_const * dist_gbest 
            self.velocity[d] = final_velocity

    def enforce_bounds(self) -> None:
        """When the position is outside of bounds, it is set in bounds. When a velocity is outside, it is set to 0"""
        for d in range(len(self.position)):
            if self.position[d] > Particle.upper_bound:
                self.position[d] = Particle.upper_bound - (self.position[d] / 1000)
                continue
            if self.position[d] < Particle.lower_bound:
                self.position[d] = Particle.lower_bound + (self.position[d] / 1000)
        for d in range(len(self.velocity)):
            if self.velocity[d] > Particle.upper_bound:
                self.velocity[d] = 0 
                continue
            if self.velocity[d] < Particle.lower_bound:
                self.velocity[d] = 0 

    def search(self) -> bool:
        """A single particle's search for minimum"""
        self.update_velocity()
        self.update_position()
        self.enforce_bounds()
        if self.update_pbest_pos():
            if self.update_gbest_pos():
                return Particle.within_target_error()
        return False

def setup_plot():
    """Based off of https://machinelearningmastery.com/a-gentle-introduction-to-particle-swarm-optimization/"""
    l = Particle.lower_bound
    u = Particle.upper_bound
    x, y = np.array(np.meshgrid(np.linspace(l,u,100), np.linspace(l,u,100)))
    z = Particle.fitness([x, y])
    x_min = x.ravel()[z.argmin()]
    y_min = y.ravel()[z.argmin()]
    fig =plt.figure(figsize=(8,6))
    plt.imshow(z, extent=[l, u, l, u], origin='lower', cmap='viridis', alpha=0.5)
    plt.colorbar()
    plt.plot([x_min], [y_min], marker='x', markersize=5, color="white")
    contours = plt.contour(x, y, z, 10, colors='black', alpha=0.4)
    plt.clabel(contours, inline=True, fontsize=8, fmt="%.0f")
    return fig
    
    
def particle_swarm_optimization(social=1.5, cognitive=1.5):
    Particle.setup(social=social, cognitive=cognitive, dimensions=2, upper=5.0, lower=-5.0, target=[0,0], error=1e-6)
    num_particles = 5
    particles = [Particle()] * num_particles
    found_target = False
    i = 0
    iterations = 50
    fig = setup_plot()
    artists = []
    plt.ioff()

    while found_target == False and i < iterations:
        for particle in particles:
            found_target = particle.search()
        Particle.decrement_weight()
        i += 1
        x_positions = [particles[n].position[0] for n in range(len(particles))]
        y_positions = [particles[n].position[1] for n in range(len(particles))]
        frame = plt.scatter(x=x_positions,y=y_positions, c='b')
        title = plt.text(-4, 5.5, f"PSO Iteration {i}, Current Gbest is {Particle.gbest_pos}")
        artists.append([frame, title])
    print(f"Gbestpos is: {Particle.gbest_pos}, in {i} iterations")
    anim = animation.ArtistAnimation(fig, artists)
    plt.show()



def main():
    particle_swarm_optimization()

if __name__ == "__main__":
    main()
