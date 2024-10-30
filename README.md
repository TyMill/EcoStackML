# EcoStackML

**EcoStackML** to zaawansowana biblioteka do generowania danych systematycznych środowiskowych z wykorzystaniem techniki ML Stacking. Biblioteka integruje różne modele uczenia maszynowego, aby uzyskać bardziej precyzyjne prognozy i lepsze odwzorowanie skomplikowanych zależności w danych środowiskowych.

## Spis Treści
- [Opis](#opis)
- [Struktura Katalogów](#struktura-katalogów)
- [Instalacja](#instalacja)
- [Użycie](#użycie)
- [Testowanie](#testowanie)
- [Przykłady](#przykłady)
- [Wkład](#wkład)
- [Licencja](#licencja)

## Opis

EcoStackML pozwala na:
- Preprocesowanie i normalizację danych środowiskowych.
- Trening i ocenę różnych modeli uczenia maszynowego.
- Integrację wyników wielu modeli za pomocą techniki stacking.
- Generowanie realistycznych prognoz i symulacji.

## Struktura Katalogów

```
EcoStackML/
├── data/
│   ├── raw/                   # Surowe dane wejściowe
│   ├── processed/             # Przetworzone dane
├── models/
│   ├── base_models/           # Implementacje podstawowych modeli (RNN, ARIMA, Regresja, Boosting)
│   ├── meta_models/           # Implementacje modeli meta do stacking
├── preprocessing/
│   ├── data_preprocessing.py  # Skrypty do wstępnej obróbki danych
├── stacking/
│   ├── stacking.py            # Implementacja procesu stacking
├── evaluation/
│   ├── evaluation_metrics.py  # Narzędzia do ewaluacji modeli
├── notebooks/
│   ├── exploratory_analysis.ipynb # Notebook do analizy eksploracyjnej
├── scripts/
│   ├── run_stacking.py        # Skrypt do uruchomienia procesu stacking
├── tests/
│   ├── test_preprocessing.py  # Testy jednostkowe dla modułu preprocessing
│   ├── test_models.py         # Testy jednostkowe dla modeli
├── README.md                  # Dokumentacja projektu
├── requirements.txt           # Lista zależności
└── setup.py                   # Skrypt instalacyjny
```


## Instalacja

1. Sklonuj repozytorium:

    ```bash
    git clone https://github.com/your-username/EcoStackML.git
    cd EcoStackML
    ```

2. Zainstaluj wymagane zależności:

    ```bash
    pip install -r requirements.txt
    ```

3. Zainstaluj bibliotekę:

    ```bash
    python setup.py install
    ```

## Użycie

1. Uruchom skrypt `run_stacking.py`:

    ```bash
    python scripts/run_stacking.py <data_file> <target_column>
    ```

    - `<data_file>`: Ścieżka do pliku CSV z danymi.
    - `<target_column>`: Nazwa kolumny z wartością docelową.


## Przykłady

### Przykładowe użycie skryptu

Przygotuj plik CSV z danymi, np. `data/raw/sample_data.csv`.

Uruchom skrypt:

```bash
python scripts/run_stacking.py data/raw/sample_data.csv target_column
```

### Analiza Eksploracyjna
Otwórz notebook exploratory_analysis.ipynb:

```bash
jupyter notebook notebooks/exploratory_analysis.ipynb
```
Wykonaj kroki analizy eksploracyjnej, aby lepiej zrozumieć dane.

## Wkład

Chętnie przyjmujemy wkład od społeczności! Aby przyczynić się do projektu:

1. Sforkuj repozytorium.
2. Stwórz nową gałąź (git checkout -b feature/nazwa-funkcjonalności).
3. Wprowadź zmiany i wykonaj commit (git commit -am 'Dodanie nowej funkcjonalności').
4. Wypchnij zmiany do gałęzi (git push origin feature/nazwa-funkcjonalności).
5. Otwórz pull request.

## Licencja

Projekt EcoStackML jest dostępny na licencji MIT. Zobacz plik LICENSE po więcej szczegółów.

EcoStackML - Tworzenie realistycznych danych środowiskowych za pomocą zaawansowanych technik ML Stacking.

## Dotychczasowi autorzy
1. Tymoteusz Miller
2. Adrianna Łobodzińska
3. Polina Kozlovska
