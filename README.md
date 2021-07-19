# gan_context_encoder
Решение тестового задания BroutonLab
Код модели взят из репозитория https://github.com/BoyuanJiang/context_encoder_pytorch

Для решения использованы:

Python 3.9.4

Pytorch 1.9.0+cu111

Tensorboard 2.5.0



1. Для запуска тренировки на данном датасете (places365) необходимо в папке 'dataset' создать директорию 'train' и загрузить в нее папку с данными (в моем случае, это папка 'places'). Тренировка запускается из папки проекта с помощью команды 

 `python train_modified.py --cuda --wtl2 0.999 --niter 20  --batchSize 64 `
 
 Параметры запуска можно изменять. 
 
2. Размер изображений в процессе тренировки изменяется на 128х128,  размер вырезаемой области находится по центру и составляет 16х16.
3. Результаты BCELoss дескриптора и генератора выгружаются в папку '\runs\experiment_1' и с помощью Tensorboard доступны графики. Для запуска Tensorboard необходимо в терминале перейти в папку проекта и запустить команду: 

`tensorboard --logdir=runs`

далее, перейти по ссылке: http://localhost:6006/

Ниже представлены графики дискриминатора и генератора.

[![Дискриминатор:](https://github.com/RivkinMikhail/gan_context_encoder/blob/main/dis_loss.svg)


[![Генератор](https://github.com/RivkinMikhail/gan_context_encoder/blob/main/gen_loss.svg)

5. Результаты тренировки в виде изображений для каждой эпохи сохраняются в папке \result\train и разбиты по директориям соответствующим изначальным образцам, образцам с вырезанным центром и восстановленным образцам.
6. Для тестирования написан скрипт для проверки на наборе изображений. Для этого в папку \dataset\val помещается папка с тестовыми изображениями и в терминале запускается команда 

  `python test.py --netG model/netG_places.pth --dataroot dataset/val --batchSize 100`
  
 Результаты  тестирования сохраняются в корне проекта в файлах под названиями 'val_real_samples.png','val_cropped_samples.png' и 'val_recon_samples.png'

Примечание: 
1. Файл 'netG_places.pth' находится по ссылке https://drive.google.com/file/d/1Y08yl0WIH-l-4L0OSxgRzrzvZXn9EmTw/view?usp=sharing и должен быть помещен в папку 'model' 
2. Логи для Tensorboard находятся по ссылке https://drive.google.com/file/d/1WnpgmZnN1sqHwavT-c-6XSDh2gBVVHcm/view?usp=sharing (заменить папку 'runs') 
