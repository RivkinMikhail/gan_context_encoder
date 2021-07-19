# gan_context_encoder
Решение тестового задания BroutonLab
Код модели взят из репозитория https://github.com/BoyuanJiang/context_encoder_pytorch

1. Для запуска тренировки на данном датасете (places365) необходимо в папке 'dataset' создать директорию 'train' и загрузить в нее папку с данными (в моем случае, это папка 'places'). Тренировка запускается из папки проекта с помощью команды 
 'python train.py --cuda --wtl2 0.999 --niter 20  --batchSize 64 '
 Параметры запуска можно изменять. 
2. Размер изображений в процессе тренировки изменяется на 128х128, однако размер вырезаемой области находится по центру и составляет 64х64.
3. Результаты BCELoss дескриптора и генератора выгружаются в папку '\runs\experiment_1' и с помощью Tensorboard доступны графики. Для запуска Tensorboard необходимо в терминале перейти в папку проекта и запустить команду: tensorboard --logdir=runs далее, перейти по ссылке: http://localhost:6006/
4. Результаты тренировки в виде изображений для каждой эпохи сохраняются в папке \result\train и разбиты по директориям соответствующим изначальным образцам, образцам с вырезанным центром и восстановленным образцам.
5. Для тестирования возможны два варианта:
  1) Тест на одном изображении. Для этого изображение помещается в папку \result\test\real и в терминале запускается команда 
  python test_one.py --netG model/netG_places.pth --test_image result/test/real/065_im.png 
  2) Тест на наборе изображений. Для этого в папку \dataset\val помещается папка с тестовыми изображениями и в терминале запускается команда 
  python test.py --netG model/netG_places.pth --dataroot dataset/val --batchSize 100
  Результаты обоих вариантов тестирования сохраняются в корне проекта в файлах под названиями 'val_real_samples','val_cropped_samples' и 'val_recon_samples'

Примечание: 
1. Файл 'netG_places.pth' находится по ссылке https://drive.google.com/file/d/1avDN3UUeyeQnj39YfqVnGSwYDH6D4a8m/view?usp=sharing и должен быть помещен в папку 'model' 
2. Логи для Tensorboard находятся по ссылке https://drive.google.com/file/d/12slIiz_5tE9vsbDQ56eWm8llCr06Gupo/view?usp=sharing (заменить папку 'runs') 
