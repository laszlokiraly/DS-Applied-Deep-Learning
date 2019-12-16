# Results

## baseline to beat

```bash
  altar    apse    bell_tower    column    dome(inner)    dome(outer)    flying_buttress    gargoyle    stained_glass    vault
-------  ------  ------------  --------  -------------  -------------  -----------------  ----------  ---------------  -------
  0.906   0.874         0.903     0.953         0.967           0.937              0.805       0.923            0.990    0.925
```

## hrnet small v1

BatchSize=128
Workers=4
Epochs=10

```bash
Epoch 10/10
----------
train Loss: 0.2350 Acc: 0.9261
test Loss: 0.3043 Acc: 0.8974

F1 Score:
  altar    apse    bell_tower    column    dome(inner)    dome(outer)    flying_buttress    gargoyle    stained_glass    vault
-------  ------  ------------  --------  -------------  -------------  -----------------  ----------  ---------------  -------
 0.9286  0.6842        0.8462    0.8512         0.9143          0.886             0.8667      0.9573           0.9693   0.9194

Training complete in 9m 28s
Best val Acc: 0.897436
```

## resnet18

BatchSize=256
Workers=4
Epochs=15

```bash
Epoch 15/15
----------
train Loss: 0.2874 Acc: 0.9139
test Loss: 0.3614 Acc: 0.8917

F1 Score:
  altar    apse    bell_tower    column    dome(inner)    dome(outer)    flying_buttress    gargoyle    stained_glass    vault
-------  ------  ------------  --------  -------------  -------------  -----------------  ----------  ---------------  -------
  0.942  0.6667        0.8653    0.8427         0.9014         0.8707             0.7571      0.9516           0.9586   0.9298

Training complete in 10m 21s
Best val Acc: 0.891738
```

## resnet152

BatchSize=32
Workers=4
Epochs=10

```bash
Epoch 9/10
----------
train Loss: 0.1310 Acc: 0.9580
test Loss: 0.1859 Acc: 0.9416

F1 Score:
  altar    apse    bell_tower    column    dome(inner)    dome(outer)    flying_buttress    gargoyle    stained_glass    vault
-------  ------  ------------  --------  -------------  -------------  -----------------  ----------  ---------------  -------
 0.9524  0.8713        0.9188    0.9423         0.9091         0.9211             0.9388      0.9852            0.951   0.9391

Training complete in 32m 57s
Best val Acc: 0.941595
```

## hrnet largest

BatchSize=20
Workers=4
Epochs=10

```bash
Epoch 9/10
----------
train Loss: 0.1109 Acc: 0.9629
test Loss: 0.1847 Acc: 0.9409

F1 Score:
  altar    apse    bell_tower    column    dome(inner)    dome(outer)    flying_buttress    gargoyle    stained_glass    vault
-------  ------  ------------  --------  -------------  -------------  -----------------  ----------  ---------------  -------
 0.9203  0.8723        0.9313    0.9317         0.9559           0.92             0.9459      0.9831           0.9658   0.9275

Training complete in 93m 37s
Best val Acc: 0.940883

```
