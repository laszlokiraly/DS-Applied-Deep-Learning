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
saving final model state to ./hrnet_final_state.pth.tar
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
saving final model state to ./resnet18_final_state.pth.tar
```
