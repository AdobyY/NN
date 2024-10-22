# Simple Neural Net

## Структура проекту

- **ReLU:** Реалізація функції активації ReLU та її похідної для використання у передній та зворотній передачі.
- **DenseLayer:** Щільний шар нейронів з випадковими вагами та зміщенням, підтримує передню та зворотню передачу.
- **MSE:** Функція середньоквадратичної помилки (MSE) для оцінки втрат.
- **NN (Neural Network):** Клас для побудови та навчання нейронної мережі з кількома шарами.
- **GDOptimizer:** Оптимізатор градієнтного спуску для оновлення ваг.

## Використання

1. **Ініціалізуйте нейронну мережу:**
    ```python
    model = NN(input_size=4)
    ```

2. **Додайте шари:**
    ```python
    model.add_layer(DenseLayer(n_units=8, activation=ReLU()))
    model.add_layer(DenseLayer(n_units=3, activation=ReLU()))
    ```

3. **Навчіть модель:**
    ```python
    optimizer = GDOptimizer(eta=0.01)
    loss_fn = MSE()
    model.fit(X_train, y_train, epochs=1000, optimizer=optimizer, loss_fn=loss_fn, verbose=True)
    ```

## Основні компоненти

- **forward:** Пропуск даних через нейронну мережу.
- **backward:** Зворотне розповсюдження градієнтів для оновлення ваг.
- **fit:** Основна функція навчання, яка використовує forward і backward передачі.

## Залежності

Для запуску проекту потрібен лише `numpy`:
```bash
pip install numpy
