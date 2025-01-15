import math
from dataclasses import dataclass
from typing import List


@dataclass
class Point:
    x: int
    y: int

    def distance(self, position) -> float:
        return math.sqrt((self.x - position.x) ** 2 + (self.y - position.y) ** 2)

    def __str__(self) -> str:
        return f"({self.x}, {self.y})"


@dataclass
class Line:
    start: Point
    end: Point

    def length(self) -> float:
        return self.start.distance(self.end)

    def __str__(self) -> str:
        return f"Line from {self.start} to {self.end}"

    def calculate_slope(self) -> float | None:
        try:
            slope = (self.end.y - self.start.y) / (self.end.x - self.start.x)
            return slope
        except ZeroDivisionError:
            print("Undefined (Vertical line)")
            return None

    def is_parallel(self, other_line: "Line") -> bool:
        if self.calculate_slope() is None and other_line.calculate_slope() is None:
            return True
        return self.calculate_slope() == other_line.calculate_slope()

    def is_perpendicular(self, other_line: "Line") -> bool:
        slope_self = self.calculate_slope()
        slope_other = other_line.calculate_slope()

        # 判斷特殊情況：一條水平線和一條垂直線
        if (slope_self is None and slope_other == 0) or (
            slope_self == 0 and slope_other is None
        ):
            return True

        # 判斷斜率相乘是否等於 -1
        if isinstance(slope_self, float) and isinstance(slope_other, float):
            return slope_self * slope_other == -1

        return False


@dataclass
class Circle:
    center: Point
    radius: int

    def area(self):
        return math.pi * self.radius**2

    def perimeter(self):
        return 2 * 3.14 * self.radius

    def __str__(self):
        return f"Circle with radius {self.radius}"

    def is_intersect(self, other_circle: "Circle") -> bool:
        center_distance = self.center.distance(other_circle.center)
        total_radius = self.radius + other_circle.radius
        radius_difference = abs(self.radius - other_circle.radius)

        return radius_difference < center_distance < total_radius


@dataclass
class Polygon:
    points: List[Point]

    def calculate_perimeter(self) -> float:
        perimeter = 0
        point_counts = len(self.points)
        for i in range(point_counts):
            perimeter += self.points[i].distance(self.points[(i + 1) % point_counts])
        return perimeter


if __name__ == "__main__":

    line_a = Line(Point(2, 4), Point(-6, 1))
    line_b = Line(Point(2, 2), Point(-6, -1))
    line_c = Line(Point(-1, 6), Point(-4, -4))
    circle_a = Circle(Point(6, 3), radius=2)
    circle_b = Circle(Point(8, 1), radius=1)
    polygon_a = Polygon(points=[Point(2, 0), Point(5, -1), Point(4, -4), Point(-1, -2)])

    print("Are Line A and Line B parallel?", line_a.is_parallel(line_b))
    print("Are Line C and Line A perpendicular?", line_a.is_perpendicular(line_c))
    print("Print the area of Circle A.", circle_a.area())
    print("Do Circle A and Circle B intersect?", circle_a.is_intersect(circle_b))
    print("Print the perimeter of Polygon A.", polygon_a.calculate_perimeter())
