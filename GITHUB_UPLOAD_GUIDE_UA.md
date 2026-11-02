# Як опублікувати цей проєкт на GitHub

## 1. Створи порожній репозиторій

Відкрий [github.com/new](https://github.com/new) і задай назву `sound-event-detection-mlpc`.

Для першого завантаження **не додавай** автоматично README, .gitignore чи LICENSE: вони вже є у нашому архіві. Постав **Private**, якщо хочеш спочатку перевірити файли; після перевірки можна змінити на Public.

## 2. Розпакуй цей ZIP і відкрий термінал у папці проєкту

На Windows: відкрий папку `sound-event-detection-portfolio`, клацни правою кнопкою та обери **Open in Terminal** (або введи `cmd` у рядок адреси Провідника).

## 3. Виконай команди

```powershell
git init
git add .
git commit -m "Add sound event detection portfolio project"
git branch -M main
git remote add origin https://github.com/volodymyrhalian/sound-event-detection-mlpc.git
git push -u origin main
```

Якщо Git запитає ім'я та email, заповни свої облікові дані (не чужі).

```powershell
git config --global user.name "Volodymyr Halian"
git config --global user.email "YOUR_GITHUB_EMAIL"
```

**Увага:** перевір `.gitignore` перед публікацією. Архів не містить повних `.npz` датасетів, оригінальних записів аудіо й персональних описів сцен. Не завантажуй їх на GitHub без перевірки ліцензій і приватності.

## 4. Перевір GitHub

Мають відображатися: `README.md`, `src/`, `tests/`, `notebooks/`, `docs/`, `checkpoints/`, `RECOVERY_NOTES.md`. У верхній частині репозиторію буде красивий опис проєкту з результатами.

Після публікації додай до свого портфоліо посилання:

`https://github.com/volodymyrhalian/sound-event-detection-mlpc`

> Результати в README взято зі звітів двох авторів, а відновлений після видалення файлів код чесно позначено як реконструкцію. Переконайся, що університет дозволяє публікацію вихідних ноутбуків, наданого starter code та партнерських матеріалів.
