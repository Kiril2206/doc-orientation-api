# Model v2: Pure CV и Hybrid

Реализация от 20.09.2026. Полноценные веса v2 ещё нужно обучить на проверенных документах. Артефакт в output/v2_smoke используется только для проверки кода, а не для исправления реальных документов.

**1. Режимы API и интерфейса**

| Параметр mode | Модель | OCR | Когда модель отсутствует |
|---|---|---|---|
| pure | model/orientation_model_v2.onnx | Не импортируется, не создаётся и не вызывается | HTTP 503 с объяснением |
| hybrid | model/orientation_model.onnx (v1) | Лениво подключается при неоднозначности 0/180 | HTTP 503, если нет v1 |

Это сравнение двух продуктовых пайплайнов, а не строгое сравнение OCR на одинаковом бэкбоне. Для такого эксперимента можно вызвать один экземпляр OrientationClassifier с mode=pure и mode=hybrid.

Настройки в app/core/config.py: inference_mode, model_path_v1, model_path_v2, inference_threads, hybrid_confidence_threshold, hybrid_margin. Старый APP_MODEL_PATH поддерживается только как переопределение v1.

~~~powershell
$env:APP_INFERENCE_MODE = "pure"
$env:APP_INFERENCE_THREADS = "2"
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
~~~

Если сервер уже работает, остановите его и запустите заново. Веб-страница получает доступность моделей из /health. Нет автоматического перехода с отсутствующей v2 на v1.

~~~powershell
curl.exe -X POST http://127.0.0.1:8000/correct-orientation -F "file=@document.pdf" -F "mode=pure" --output corrected.pdf
curl.exe -X POST http://127.0.0.1:8000/correct-orientation -F "file=@document.png" -F "mode=hybrid" --output corrected.png
~~~

Явный mode имеет приоритет над настройкой сервера. При отсутствии поля используется APP_INFERENCE_MODE (по умолчанию pure). Неизвестное значение возвращает 422. Для PDF выбранный mode передаётся каждой странице. Инференс выполняется в рабочем потоке, чтобы не блокировать event loop.

Hybrid вызывает OCR только если два наиболее вероятных класса — 0 и 180, и уверенность CNN ниже 0.90 либо разрыв их вероятностей меньше 0.25. Эти пороги — исходная эвристика, их следует подобрать на validation. Уверенная ошибка CNN может пройти без OCR. OCR может выбрать только между 0 и 180, а не поменять ось документа.

Заголовки ответа:
- X-Inference-Mode: фактический режим.
- X-Model-Version: v1 или v2.
- X-Decision-Source: cv, cv_no_ambiguity, ocr_override, cv_ocr_confirmed, cv_ocr_unavailable, cv_ocr_inconclusive или cv_ocr_error.
- X-Decision-Counts: количество страниц по источнику решения; для PDF общий X-Decision-Source может быть mixed.
- X-Confidence: вероятность итогового угла по CNN, а не выдуманная уверенность OCR. При OCR-переопределении она может быть низкой; для PDF передаётся минимум.
- X-Original-Orientation и X-Rotation-Applied: по часовой стрелке; для PDF описывают первую страницу.

Интерфейс показывает переключатель и плашку пайплайна. Сервер пишет в лог mode, model и decision. PDF сохраняет текст и исходное содержимое, меняется только поворот страницы.

Pure не требует установки OCR:

~~~powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
~~~

Для демонстрации Hybrid:

~~~powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-hybrid.txt
~~~

При недоступном OCR сохраняется решение CNN с явным статусом cv_ocr_unavailable. Docker по умолчанию собирается без OCR; для Hybrid используйте docker compose build --build-arg INSTALL_HYBRID=true, затем docker compose up.

**2. Архитектура**

По умолчанию: EfficientNet-B0 + линейная голова на 4 класса, ImageNet-инициализация, вход 384×384.

| Бэкбон | Параметры исходной ImageNet-модели, приблизительно | Роль |
|---|---:|---|
| EfficientNet-B0 | 5.3 млн | Первый кандидат для ограниченного CPU |
| ResNet-34 | 21.8 млн | Простой сравнительный baseline, существенно больше вычислений |
| ResNet-50 | 25.6 млн | Кандидат, если прирост качества оправдает задержку |
| ConvNeXt-Tiny | 28.6 млн | Современный CNN, но тяжелее B0; нужно измерять ONNX |

После замены 1000-классовой головы количество параметров уменьшается. Эти числа не предсказывают документную точность или задержку. Источники: официальные страницы [B0](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_b0.html), [ResNet-34](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet34.html), [ResNet-50](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html), [ConvNeXt-Tiny](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.convnext_tiny.html).

384 выбрано как начальный компромисс. Поддерживается 448, но его необходимо сравнить на том же test split и CPU. Разрешение 384 не гарантирует различимость мелкого текста на любой странице: для длинных чеков и очень плотных страниц далее могут понадобиться несколько детальных фрагментов.

Предобработка общая для train/inference: EXIF → RGB → сохранение пропорций → белые поля → ImageNet-нормализация. Полная страница не обрезается центральным crop. V1 сохраняет свой исходный resize 224×224, чтобы не менять контракт имеющихся весов.

Метки checkpoint сохраняют прежний порядок: [0, 90, 180, 270] ПРОТИВ часовой стрелки. API переводит их в [0, 270, 180, 90] ПО часовой стрелке и применяет обратный поворот.

**3. Локальные примеры и разделение**

Создайте data/local.jsonl. Каждая строка — JSON-объект. Пути задаются относительно файла манифеста:

~~~json
{"path":"local/passport_ru_01.jpg","group_id":"passport-family-01","split":"train","upright_verified":true,"language":"ru","category":"passport"}
{"path":"local/certificate_ru_02.pdf","group_id":"certificate-family-02","split":"validation","upright_verified":true,"language":"ru","category":"certificate"}
{"path":"local/form_multilingual_03.png","group_id":"form-family-03","split":"test","upright_verified":true,"language":"ru-en","category":"form"}
~~~

Это пример структуры, не достаточный объём обучения. Добавьте разные документы, языки, способы съёмки и шаблоны. Для PDF без page будут подготовлены все страницы; page задаёт индекс с нуля. correction_cw позволяет указать известное исправление 0/90/180/270. upright_verified означает, что страница проверена с учётом этого исправления.

Один group_id должен объединять страницы одного документа, дубликаты и тесно связанные варианты шаблона. Все они должны быть в одном split. Если split не задан, используется стабильное разбиение по group_id. Для небольшого корпуса задавайте splits явно, чтобы validation/test не оказались пустыми.

Код проверяет пересечение групп между splits, повторные экспорты одинаковых файлов и контрольные суммы. Он не заменяет поиск визуально близких дубликатов. Кириллица и паспорта не появятся автоматически из ImageNet/DocLayNet: их качество зависит от представительности ваших локальных примеров.

**4. Подготовка RGB-корпуса один раз**

~~~powershell
.\.venv\Scripts\python.exe -u -m training.prepare_data --output-dir data/v2 --doclaynet 5000 --cord 600 --local-manifest data/local.jsonl
~~~

Скрипт:
- Использует официальный [DocLayNet v1.2](https://huggingface.co/datasets/docling-project/DocLayNet-v1.2) и [CORD v2](https://huggingface.co/datasets/naver-clova-ix/cord-v2).
- Фиксирует commit revision каждого источника.
- Читает только image и необходимые метаданные, исключая PDF binary payload.
- Сохраняет RGB PNG с длинной стороной до 1024, манифест и контрольные суммы.
- Сохраняет официальный train/validation/test для удалённых источников и группы локальных документов.
- Пишет манифест после каждой страницы. При повторе завершённые источники берутся с диска; незавершённый источник перечитывается с пропуском уже записанных страниц.
- Не использует удалённые источники во время последующих эпох обучения.

DocLayNet даёт разнообразные страницы; CORD добавляет чеки, но его Parquet-файлы могут долго скачиваться. В случае сбоев CORD можно начать отдельный корпус с --cord 0, не смешивая его параметры с уже созданным каталогом. Изменение параметров подготовки требует другого --output-dir. Скрипт не публикует локальные документы в облако.

**5. Необязательный аудит вертикальности**

~~~powershell
.\.venv\Scripts\python.exe -m training.review_data --manifest data/v2/manifest.jsonl
~~~

Страницы DocLayNet и CORD, скачанные через prepare_data, сразу считаются вертикальными и получают upright_verified=true. Поэтому этот шаг не требуется для запуска обучения. При желании откройте data/v2/review.html, чтобы выборочно проверить страницы, исправить поворот или исключить пустые и неоднозначные примеры.

Это доверие к официальным корпусам является осознанным допущением: редкая неверно ориентированная исходная страница станет шумной меткой. Локальные файлы по-прежнему требуют upright_verified=true. Параметр --allow-unreviewed оставлен только для сторонних манифестов.

**6. Обучение v2**

Зависимости для обучения:

~~~powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
~~~

На GTX 1050 Ti 4 GB начните с batch-size 8; при нехватке VRAM используйте 4.

~~~powershell
.\.venv\Scripts\python.exe -u -m training.train --manifest data/v2/manifest.jsonl --backbone efficientnet_b0 --input-size 384 --epochs 20 --batch-size 8 --device cuda --output-dir output/v2
~~~

ImageNet-веса используются по умолчанию. --no-pretrained предназначен для проверки кода или контролируемого обучения с нуля. Старая v1 не используется для инициализации B0.

Каждая базовая страница даёт ровно четыре угла. Аугментации включают ColorJitter, небольшой RandomAffine, плавные тени, GaussianBlur и слабый шум. Отражений и неконтролируемых поворотов на 180° нет. Validation/test проходят только детерминированную предобработку.

Если доля локальных паспортов мала, задайте смесь источников:
~~~powershell
.\.venv\Scripts\python.exe -u -m training.train --manifest data/v2/manifest.jsonl --epochs 20 --batch-size 8 --device cuda --source-weights examples/source_weights.json --output-dir output/v2_balanced
~~~

Названия должны точно соответствовать представленным training-источникам. Семплер выбирает базовые страницы по источникам и выпускает все четыре угла каждой выбранной страницы, сохраняя баланс ориентаций. Validation/test не пересэмплируются.

Сохраняются:
- best_model.pth: лучший по validation macro F1 checkpoint с архитектурой и контрактом входа.
- last_checkpoint.pth: веса, optimizer, scheduler, AMP scaler, эпоха, история и RNG для продолжения.
- training_history.json: loss, accuracy, macro F1, Precision/Recall/F1/support каждого угла, confusion matrix и число ошибок 0↔180.
- training_config.json: параметры и SHA-256 манифеста.

Продолжение после остановки:

~~~powershell
.\.venv\Scripts\python.exe -u -m training.train --manifest data/v2/manifest.jsonl --backbone efficientnet_b0 --input-size 384 --epochs 20 --batch-size 8 --device cuda --output-dir output/v2 --resume output/v2/last_checkpoint.pth
~~~

Используйте те же параметры исходного запуска, включая source-weights, если они были. Восстанавливается последняя завершённая эпоха; незавершённая выполняется заново. Изменять манифест во время возобновления нельзя.

**7. Проверка качества**

~~~powershell
.\.venv\Scripts\python.exe -m training.evaluate --model-path output/v2/best_model.pth --manifest data/v2/manifest.jsonl --batch-size 8 --device cuda --output-dir output/v2/evaluation
~~~

Оценивается только test split без OCR. В отчёте есть Precision/Recall для 0/180 отдельно, confusion matrix и разрезы source/language/category. Не подбирайте параметры по test: используйте validation, затем однократно проверяйте финального кандидата. Для сравнения бэкбонов используйте один и тот же манифест, seed и протокол.

Отдельно проверьте реальные паспорта/обложки, длинные чеки, формы, фотографии и смешанные PDF. Уверенность softmax не равна гарантии корректного поворота. Пустые и симметричные страницы принципиально неоднозначны; текущий API не реализует откалиброванный abstain.

**8. Экспорт и CPU-проверка**

~~~powershell
.\.venv\Scripts\python.exe -m training.export_onnx --model-path output/v2/best_model.pth --output-path model/orientation_model_v2.onnx --verify
.\.venv\Scripts\python.exe -m training.benchmark --model-path model/orientation_model_v2.onnx --threads 2 --runs 100 --target-ms 100
~~~

Экспорт берёт бэкбон и разрешение из checkpoint, сохраняет метаданные в ONNX, проверяет parity для batch=1 и batch=2 и только после успеха заменяет файл назначения. Простое изменение размера входа старой v1 не превращает её в v2.

Замер от 20.09.2026 на контрольном B0 384, ONNX Runtime 1.30.0, Windows CPU Family 6 Model 158:
- 1 поток: median 79.1 мс, p95 88.9 мс, max 125.9 мс.
- 2 потока: median 55.5 мс, p95 59.3 мс, max 60.7 мс.
- 50 запросов после 10 прогревочных; RGB-страница 1240×1754, batch=1.
- Время включает resize/normalization/ONNX; исключает decode, PDF rendering, HTTP и OCR.
- Использованы контрольные веса, обученные одну эпоху на четырёх синтетических страницах. Это свидетельство работоспособности и задержки архитектуры, а не её точности.

Значения нужно повторить на финальных весах и целевом CPU, в том числе под параллельной нагрузкой. Условие benchmark — p95 < 100 мс; гарантии для каждого полного HTTP/PDF-запроса оно не даёт. APP_INFERENCE_THREADS=2 выбран по этому локальному замеру; для 1-vCPU сервера поставьте 1 и измерьте заново.

После экспорта перезапустите API. Pure станет доступен, Hybrid продолжит использовать прежний файл v1. Основной рабочий v1 не удаляется.
