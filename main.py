import numpy as np
import matplotlib.pyplot as plt


class Sphere:
    def __init__(
        self, center, radius, ambient, diffuse, specular, shininess, reflection
    ):
        self.center = np.array(center)
        self.radius = radius
        self.ambient = np.array(ambient)
        self.diffuse = np.array(diffuse)
        self.specular = np.array(specular)
        self.shininess = shininess
        self.reflection = reflection

    def intersect(self, ray_origin, ray_direction):
        b = 2 * np.dot(ray_direction, ray_origin - self.center)
        c = np.linalg.norm(ray_origin - self.center) ** 2 - self.radius**2
        delta = b**2 - 4 * c
        if delta > 0:
            t1 = (-b + np.sqrt(delta)) / 2
            t2 = (-b - np.sqrt(delta)) / 2
            if t1 > 0 and t2 > 0:
                return min(t1, t2)
        return None


def reflected(vector, axis):
    return vector - 2 * np.dot(vector, axis) * axis


def normalize(vector):
    return vector / np.linalg.norm(vector)


def closest_object(objects, ray_origin, ray_direction):
    distances = [obj.intersect(ray_origin, ray_direction) for obj in objects]
    min_distance = np.inf
    closest_obj = None
    for obj, distance in zip(objects, distances):
        if distance and distance < min_distance:
            min_distance = distance
            closest_obj = obj
    return closest_obj, min_distance


def compute_lighting(point, normal, view, objects, light, closest_obj):
    illumination = np.zeros(3)

    # Ambient
    illumination += closest_obj.ambient * light["ambient"]

    # Diffuse
    light_dir = normalize(light["position"] - point)
    _, shadow_distance = closest_object(objects, point + 1e-5 * normal, light_dir)
    light_distance = np.linalg.norm(light["position"] - point)
    if shadow_distance < light_distance:
        return illumination

    illumination += (
        closest_obj.diffuse * light["diffuse"] * max(np.dot(light_dir, normal), 0)
    )

    # Specular
    H = normalize(light_dir + view)
    illumination += (
        closest_obj.specular
        * light["specular"]
        * max(np.dot(normal, H), 0) ** (closest_obj.shininess / 4)
    )

    return illumination


def render_scene(camera, objects, light, screen, width, height, max_depth):
    image = np.zeros((height, width, 3))
    aspect_ratio = width / height
    for i, y in enumerate(np.linspace(screen[1], screen[3], height)):
        for j, x in enumerate(np.linspace(screen[0], screen[2], width)):
            pixel = np.array([x, y, 0])
            origin = camera
            direction = normalize(pixel - origin)
            color = np.zeros(3)
            reflection = 1

            for _ in range(max_depth):
                closest_obj, min_distance = closest_object(objects, origin, direction)
                if closest_obj is None:
                    break

                intersection = origin + min_distance * direction
                normal = normalize(intersection - closest_obj.center)
                shifted_point = intersection + 1e-5 * normal
                view = normalize(camera - intersection)

                lighting = compute_lighting(
                    shifted_point, normal, view, objects, light, closest_obj
                )
                color += reflection * lighting
                reflection *= closest_obj.reflection

                origin = shifted_point
                direction = reflected(direction, normal)

            image[i, j] = np.clip(color, 0, 1)
        print(f"Rendering progress: {i + 1}/{height}")

    return image


if __name__ == "__main__":
    width = 400
    height = 300
    max_depth = 3

    camera = np.array([0, 0, 1])
    screen = (-1, 1 / (width / height), 1, -1 / (width / height))
    light = {
        "position": np.array([5, 5, 5]),
        "ambient": np.array([1, 1, 1]),
        "diffuse": np.array([1, 1, 1]),
        "specular": np.array([1, 1, 1]),
    }

    objects = [
        Sphere(
            [-0.5, 0, -1.5], 0.4, [0.1, 0, 0], [0.7, 0, 0], [1, 1, 1], 50, 0.5
        ),  # Red sphere
        Sphere(
            [0.6, -0.3, -1], 0.2, [0, 0.1, 0], [0, 0.7, 0], [1, 1, 1], 100, 0.4
        ),  # Green sphere
        Sphere(
            [-0.2, 0.5, -0.8], 0.1, [0, 0, 0.1], [0, 0, 0.7], [1, 1, 1], 100, 0.3
        ),  # Blue sphere
        Sphere(
            [0.3, 0.2, -0.5], 0.3, [0.1, 0.1, 0], [0.7, 0.7, 0], [1, 1, 1], 150, 0.6
        ),  # Yellow sphere
        Sphere(
            [0, -9000, 0],
            9000 - 0.7,
            [0.1, 0.1, 0.1],
            [0.6, 0.6, 0.6],
            [1, 1, 1],
            100,
            0.3,
        ),  # Large ground sphere
    ]

    image = render_scene(camera, objects, light, screen, width, height, max_depth)
    plt.imsave("scene.png", image)
