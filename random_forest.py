"""
Clasificación con Random Forest usando Scikit-learn
Dataset: UCI Wine Dataset

Este proyecto implementa un modelo Random Forest utilizando
la biblioteca Scikit-learn para clasificar vinos en tres clases.
"""

import csv
from pathlib import Path

from sklearn.model_selection import train_test_split


# Ruta al dataset
DATASET_PATH = Path(__file__).with_name("wine.data")


def cargar_datos(file_path):
    """
    Lee el UCI Wine Dataset.

    La primera columna contiene la clase del vino.
    Las 13 columnas restantes contienen los features.
    """

    features = []
    labels = []

    with open(file_path, "r", encoding="utf-8") as file:
        reader = csv.reader(file)

        for row in reader:
            if not row:
                continue

            label = int(row[0])
            sample = [float(value) for value in row[1:]]

            labels.append(label)
            features.append(sample)

    return features, labels


def contar_clases(labels):
    """
    Cuenta cuántas observaciones pertenecen a cada clase.
    """

    conteo = {}

    for label in labels:
        conteo[label] = conteo.get(label, 0) + 1

    return conteo


def main():
    """Ejecuta la carga y división inicial del dataset."""

    # ----------------------------------------------------------
    # 1. Cargar datos
    # ----------------------------------------------------------

    features, labels = cargar_datos(DATASET_PATH)

    print("UCI Wine Dataset")
    print("----------------")
    print(f"Número de observaciones: {len(features)}")
    print(f"Número de features: {len(features[0])}")
    print(f"Clases encontradas: {sorted(set(labels))}")

    # ----------------------------------------------------------
    # 2. Separar training del resto de los datos
    # ----------------------------------------------------------

    (
        x_train,
        x_temp,
        y_train,
        y_temp
    ) = train_test_split(
        features,
        labels,
        test_size=0.30,
        random_state=42,
        stratify=labels
    )

    # ----------------------------------------------------------
    # 3. Dividir el 30% restante entre validation y test
    # ----------------------------------------------------------

    (
        x_validation,
        x_test,
        y_validation,
        y_test
    ) = train_test_split(
        x_temp,
        y_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp
    )

    print("\nDivisión del dataset")
    print("--------------------")
    print(f"Training:   {len(x_train)} observaciones")
    print(f"Validation: {len(x_validation)} observaciones")
    print(f"Test:       {len(x_test)} observaciones")

    print("\nDistribución de clases")
    print("----------------------")
    print(f"Training:   {contar_clases(y_train)}")
    print(f"Validation: {contar_clases(y_validation)}")
    print(f"Test:       {contar_clases(y_test)}")


if __name__ == "__main__":
    main()
