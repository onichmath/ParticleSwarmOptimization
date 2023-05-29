import numpy as np 
from matplotlib import pyplot as plt

class Particle():
    """Represents a particle"""
    dimensions = 2
    gbest_pos = [1000] * dimensions
    # Inertial weight
    weight = 0.8 
    # Social and cognitive coefficients.
    # Cognitive makes a particle care more about its own findings
    # Social makes a particle care more about the swarm's findings
    social_coefficient = 1.5 
    cognitive_coefficient = 1.5
    # Upper and lower bounds of the problem
    upper_bound = 5
    lower_bound = -5

    @classmethod
    def reset_dimensions(cls, dimensions):
        """Reset the dimensions and gbest_pos"""
        cls.dimensions = dimensions
        cls.gbest_pos = [9999] * dimensions

    @classmethod
    def decrement_weight(cls):
        """Decrement the weight/inertia of all particles"""
        if cls.weight > 0.3:
            cls.weight = cls.weight - 0.01

    @classmethod
    def set_coefficients(cls, social, cognitive) -> None:
        """Sets social and cognitive coefficients"""
        cls.social_coefficient = social
        cls.cognitive_coefficient = cognitive

    @classmethod
    def set_bounds(cls, upper, lower) -> None:
        """Sets the bounds of the problem space"""
        cls.upper_bound = upper
        cls.lower_bound = lower

    @classmethod
    def fitness(cls, position):
        """Ackley"""
        x = position[0]
        y = position[1]
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

    def update_gbest_pos(self) -> None:
        """Updates gbest based on particle index"""
        if self.fitness(self.pbest_pos) < self.fitness(Particle.gbest_pos):
            Particle.gbest_pos = self.pbest_pos 

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
            p_const = Particle.cognitive_coefficient * np.random.uniform(low=0, high=1)
            # Set the social constant
            g_const = Particle.social_coefficient * np.random.uniform(low=0, high=1)
            # Set velocity in given dimension
            final_velocity = inertial_velocity + p_const * dist_pbest + g_const * dist_gbest 
            self.velocity[d] = final_velocity

    def enforce_bounds(self) -> None:
        """When the position is outside of bounds, it is set in bounds. When a velocity is outside, it is set to 0"""
        for d in range(len(self.position)):
            if self.position[d] > Particle.upper_bound:
                self.position[d] = Particle.upper_bound
                continue
            if self.position[d] < Particle.lower_bound:
                self.position[d] = Particle.upper_bound
        for d in range(len(self.velocity)):
            if self.velocity[d] > Particle.upper_bound:
                self.velocity[d] = 0
                continue
            if self.velocity[d] < Particle.lower_bound:
                self.velocity[d] = 0

    def search(self):
        """A single particle's search for minimum"""
        self.update_velocity()
        self.update_position()
        self.enforce_bounds()
        if self.update_pbest_pos():
            self.update_gbest_pos()

def particle_swarm_optimization():
    num_particles = 50
    particles = [Particle()] * num_particles
    i = 500
    while i > 0:
        for particle in particles:
            print(particle)
            particle.search()
        Particle.decrement_weight()
        i -= 1
    print(f"Gbestpos is: {Particle.gbest_pos}")




def main():
    particle_swarm_optimization()

if __name__ == "__main__":
    main()
