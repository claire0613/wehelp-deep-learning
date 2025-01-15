import math
from dataclasses import dataclass
from typing import List


@dataclass
class Vector:
    x: int
    y: int


class Position:
    def __init__(self, x: int, y: int) -> None:
        self._x = x
        self._y = y

    @property
    def x(self) -> int:
        return self._x

    @property
    def y(self) -> int:
        return self._y

    def move(self, vector: Vector) -> None:
        self._x += vector.x
        self._y += vector.y


class Enemy:
    def __init__(self, name: str, position: Position, movement: Vector) -> None:
        self._name = name
        self._position = position
        self._movement = movement
        self._health = 10

    @property
    def name(self):
        return self._name

    @property
    def position(self) -> Position:
        return self._position

    @property
    def health(self) -> int:
        return self._health

    def move(self) -> None:
        if self._health > 0:
            self._position.move(self._movement)

    def take_damage(self, damage: int) -> None:
        self._health = max(0, self._health - damage)

    def is_alive(self) -> bool:
        return self._health > 0

    def status(self):
        if self.is_alive():
            print(
                f"Enemy {self._name} is alive at ({self._position.x}, {self._position.y}) with {self._health} HP."
            )
        else:
            print(
                f"Enemy {self._name} is dead at ({self._position.x}, {self._position.y})."
            )


class Tower:
    def __init__(
        self,
        name: str,
        position: Position,
        attack_power: int = 1,
        attack_range: int = 2,
    ):
        self._name = name
        self._position = position
        self._attack_power = attack_power
        self._attack_range = attack_range

    @property
    def name(self):
        return self._name

    def calculate_distance(self, target: Position) -> float:
        return math.sqrt(
            (self._position.x - target.x) ** 2 + (self._position.y - target.y) ** 2
        )

    def is_in_range(self, enemy: Enemy) -> bool:
        return self.calculate_distance(enemy.position) <= self._attack_range

    def attack(self, enemy: Enemy):
        if enemy.is_alive():
            enemy.take_damage(self._attack_power)

    def take_turn(self, enemies: List[Enemy]) -> None:
        for enemy in enemies:
            if self.is_in_range(enemy):
                self.attack(enemy)


class AdvancedTower(Tower):
    def __init__(self, name: str, position: Position):
        super().__init__(name, position, attack_power=2, attack_range=4)


class Game:
    def __init__(self, rounds: int, enemies: List[Enemy], towers: List[Tower]):
        self._rounds = rounds
        self._enemies = enemies
        self._towers = towers

    def play_round(self):
        for enemy in self._enemies:
            enemy.move()

        for tower in self._towers:
            tower.take_turn(self._enemies)

    def show_status(self):
        for enemy in self._enemies:
            enemy.status()

    def start(self):
        while self._rounds > 0:
            self.play_round()
            self._rounds -= 1

        self.show_status()


if __name__ == "__main__":
    enemies = [
        Enemy("E1", Position(-10, 2), Vector(2, -1)),
        Enemy("E2", Position(-8, 0), Vector(3, 1)),
        Enemy("E3", Position(-9, -1), Vector(3, 0)),
    ]
    towers = [
        Tower("T1", Position(-3, 2)),
        Tower("T2", Position(-1, -2)),
        Tower("T3", Position(4, 2)),
        Tower("T4", Position(7, 0)),
        AdvancedTower("A1", Position(1, 1)),
        AdvancedTower("A2", Position(4, -3)),
    ]
    game = Game(10, enemies, towers)
    game.start()
