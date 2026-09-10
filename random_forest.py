"""
Clasificación de dígitos con Random Forest usando Scikit-learn.

Dataset: Digits Dataset de Scikit-learn.

El objetivo es clasificar imágenes de dígitos escritos a mano
en una de diez clases posibles: 0, 1, 2, ..., 9.
"""

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


def contar_clases(labels):
    """
    Cuenta cuántas observaciones pertenecen a cada clase.
    """

    conteo = {}

    for label in labels:
        conteo[int(label)] = conteo.get(int(label), 0) + 1

    return conteo


def main():
    """
    Carga el Digits Dataset y lo divide en conjuntos
    de training, validation y test.
    """

    # ----------------------------------------------------------
    # 1. Cargar el Digits Dataset
    # ----------------------------------------------------------

    digits = load_digits()

    features = digits.data
    labels = digits.target

    print("Scikit-learn Digits Dataset")
    print("---------------------------")
    print(f"Número de observaciones: {len(features)}")
    print(f"Número de features: {features.shape[1]}")
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
    # 3. Dividir el 30% restante en validation y test
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
