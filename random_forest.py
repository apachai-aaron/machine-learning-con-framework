"""
Clasificación de dígitos con Random Forest usando Scikit-learn.

Dataset: Digits Dataset de Scikit-learn.

El objetivo es clasificar imágenes de dígitos escritos a mano
en una de diez clases posibles: 0, 1, 2, ..., 9.
"""

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)


def contar_clases(labels):
    """
    Cuenta cuántas observaciones pertenecen a cada clase.
    """

    conteo = {}

    for label in labels:
        conteo[int(label)] = conteo.get(int(label), 0) + 1

    return conteo

def seleccionar_modelo(
    x_train,
    y_train,
    x_validation,
    y_validation
):
    """
    Prueba diferentes configuraciones de Random Forest
    y selecciona la de mayor Macro F1 en validation.
    """

    numeros_arboles = [50, 100, 200]
    profundidades = [5, 10, None]

    mejor_modelo = None
    mejor_configuracion = None
    mejor_f1 = -1.0

    resultados = []

    for n_estimators in numeros_arboles:

        for max_depth in profundidades:

            modelo = RandomForestClassifier(
                n_estimators=n_estimators,
                criterion="gini",
                max_depth=max_depth,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features="sqrt",
                bootstrap=True,
                random_state=42,
                n_jobs=-1
            )

            # Entrenar el modelo
            modelo.fit(
                x_train,
                y_train
            )

            # Predicciones sobre training
            pred_train = modelo.predict(
                x_train
            )

            # Predicciones sobre validation
            pred_validation = modelo.predict(
                x_validation
            )

            # Accuracy en ambos conjuntos
            train_accuracy = accuracy_score(
                y_train,
                pred_train
            )

            validation_accuracy = accuracy_score(
                y_validation,
                pred_validation
            )

            train_f1 = f1_score(
                y_train,
                pred_train,
                average="macro"
            )

            validation_f1 = f1_score(
                y_validation,
                pred_validation,
                average="macro"
            )

            resultados.append(
                (
                    n_estimators,
                    max_depth,
                    train_accuracy,
                    validation_accuracy,
                    train_f1,
                    validation_f1
                )
            )

            # Seleccionar usando únicamente validation
            if validation_f1 > mejor_f1:
                mejor_f1 = validation_f1
                mejor_modelo = modelo

                mejor_configuracion = {
                    "n_estimators": n_estimators,
                    "max_depth": max_depth
                }

    return (
        mejor_modelo,
        mejor_configuracion,
        mejor_f1,
        resultados
    )


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

    # ----------------------------------------------------------
    # 4. Selección de hiperparámetros
    # ----------------------------------------------------------

    print("\nSelección de hiperparámetros")
    print("----------------------------")

    (
        mejor_modelo,
        mejor_configuracion,
        mejor_f1,
        resultados
    ) = seleccionar_modelo(
        x_train,
        y_train,
        x_validation,
        y_validation
    )

    for (
        n_estimators,
        max_depth,
        train_accuracy,
        validation_accuracy,
        train_f1,
        validation_f1
    ) in resultados:

        print(
            f"Árboles: {n_estimators:3d} | "
            f"Profundidad: {str(max_depth):>4} | "
            f"Train Acc: {train_accuracy:.4f} | "
            f"Val Acc: {validation_accuracy:.4f} | "
            f"Val F1: {validation_f1:.4f}"
        )

    print("\nMejor configuración")
    print("-------------------")

    print(
        f"n_estimators: "
        f"{mejor_configuracion['n_estimators']}"
    )

    print(
        f"max_depth: "
        f"{mejor_configuracion['max_depth']}"
    )

    print(
        f"Validation Macro F1: "
        f"{mejor_f1:.4f}"
    )

    # ----------------------------------------------------------
    # 5. Entrenamiento del modelo final
    # ----------------------------------------------------------

    print("\nEntrenamiento del modelo final")
    print("------------------------------")

    modelo_final = RandomForestClassifier(
        n_estimators=mejor_configuracion["n_estimators"],
        criterion="gini",
        max_depth=mejor_configuracion["max_depth"],
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    )

    # Utilizar training + validation para entrenar
    # el modelo final con más información disponible.
    x_final_train = list(x_train) + list(x_validation)
    y_final_train = list(y_train) + list(y_validation)

    modelo_final.fit(
        x_final_train,
        y_final_train
    )

    print(
        f"Observaciones usadas para entrenamiento final: "
        f"{len(x_final_train)}"
    )

    print("Modelo final entrenado correctamente.")

    # ----------------------------------------------------------
    # 6. Predicciones sobre el test set
    # ----------------------------------------------------------

    predicciones_test = modelo_final.predict(
        x_test
    )

    print("\nPrimeras 10 predicciones")
    print("------------------------")

    for i in range(10):
        print(
            f"Ejemplo {i + 1}: "
            f"Real = {int(y_test[i])} | "
            f"Predicción = {int(predicciones_test[i])}"
        )

    # ----------------------------------------------------------
    # 7. Matriz de confusión
    # ----------------------------------------------------------

    matriz = confusion_matrix(
        y_test,
        predicciones_test
    )

    print("\nMatriz de confusión")
    print("-------------------")
    print(matriz)

    # ----------------------------------------------------------
    # 8. Métricas de evaluación
    # ----------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predicciones_test
    )

    macro_precision = precision_score(
        y_test,
        predicciones_test,
        average="macro"
    )

    macro_recall = recall_score(
        y_test,
        predicciones_test,
        average="macro"
    )

    macro_f1 = f1_score(
        y_test,
        predicciones_test,
        average="macro"
    )

    print("\nMétricas finales")
    print("----------------")

    print(
        f"Accuracy:        "
        f"{accuracy:.4f} "
        f"({accuracy * 100:.2f}%)"
    )

    print(
        f"Macro Precision: "
        f"{macro_precision:.4f}"
    )

    print(
        f"Macro Recall:    "
        f"{macro_recall:.4f}"
    )

    print(
        f"Macro F1-score:  "
        f"{macro_f1:.4f}"
    )

    print("\nReporte por clase")
    print("-----------------")

    print(
        classification_report(
            y_test,
            predicciones_test,
            digits=4
        )
    )


if __name__ == "__main__":
    main()
