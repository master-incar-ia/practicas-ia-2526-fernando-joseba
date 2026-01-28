# Este es un ejemplo de una clase simple en Python que representa una brocha para pintar.

class Brocha:
    def __init__(self, color: str, tamanio: int):
        self.color = color
        self.tamanio = tamanio

    def paint(self) -> str:
        print(f"Painting with a {self.tamanio} cm {self.color} brush.")
    
    def guardar_dibujo(self, nombre_archivo: str):
        print(f"Saving drawing as {nombre_archivo}.")

if __name__ == "__main__":
    brocha_azul = Brocha("azul", 5)
    brocha_verde = Brocha("verde", 10)
    brocha_azul_pequenia = Brocha("azul", 2)

    brocha_azul.paint()