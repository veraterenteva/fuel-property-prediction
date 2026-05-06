import random


class FuelModel:
    def __init__(self):
        # тут потом загрузим реальные модели, либо из файликов, либо как
        pass

    def get_properties_from_mixture(self, smiles=None, blend=None):
        """
        smiles: str
        blend: list of {"smiles": str, "fraction": float}
        """

        # пока стоят заглушки из случайных чисел
        return {
            "octane_number": round(random.uniform(70, 110), 2),
            "cetane_number": round(random.uniform(20, 70), 2),
            "flash_point": round(random.uniform(-20, 200), 2)
        }

    def get_mixture_from_properties(self, octane, cetane, flash_point): # inverse задача
        """
        вход: целевые свойства
        выход: смесь
        """

        # пока опять же случайные числа
        components = ["CCO", "CCCC", "CCN", "CCCl"]

        blend = []
        remaining = 1.0

        for i in range(2):
            frac = round(random.uniform(0.1, remaining), 2)
            remaining -= frac

            blend.append({
                "smiles": random.choice(components),
                "fraction": frac
            })

        blend.append({
            "smiles": random.choice(components),
            "fraction": round(remaining, 2)
        })

        return {
            "target": {
                "octane": octane,
                "cetane": cetane,
                "flash_point": flash_point
            },
            "blend": blend
        }


# singleton (важно для сервера)
model = FuelModel()