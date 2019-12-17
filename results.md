# Results

## baseline to beat

```bash
  altar    apse    bell_tower    column    dome(inner)    dome(outer)    flying_buttress    gargoyle    stained_glass    vault
-------  ------  ------------  --------  -------------  -------------  -----------------  ----------  ---------------  -------
  0.906   0.874         0.903     0.953         0.967           0.937              0.805       0.923            0.990    0.925
```

## hrnet v1 small

BatchSize=128
Workers=4
Epochs=25

```bash
Epoch 18/25
----------
train Loss: 0.2317 Acc: 0.9257
test Loss: 0.2990 Acc: 0.9017

F1 Score:
  altar    apse    bell_tower    column    dome(inner)    dome(outer)    flying_buttress    gargoyle    stained_glass    vault
-------  ------  ------------  --------  -------------  -------------  -----------------  ----------  ---------------  -------
 0.9353   0.713        0.8521     0.866          0.922         0.9043             0.8442      0.9437           0.9728   0.9254

Epoch 25/25
----------
train Loss: 0.2265 Acc: 0.9321
test Loss: 0.3011 Acc: 0.9017

F1 Score:
  altar    apse    bell_tower    column    dome(inner)    dome(outer)    flying_buttress    gargoyle    stained_glass    vault
-------  ------  ------------  --------  -------------  -------------  -----------------  ----------  ---------------  -------
 0.9353     0.7        0.8555    0.8564          0.922         0.9037             0.8497      0.9483           0.9797   0.9249

Training complete in 23m 6s
Best val Acc: 0.901709
```

## resnet18

BatchSize=256
Workers=4
Epochs=25

```bash
Epoch 23/25
----------
train Loss: 0.2837 Acc: 0.9111
test Loss: 0.3601 Acc: 0.8932

F1 Score:
  altar    apse    bell_tower    column    dome(inner)    dome(outer)    flying_buttress    gargoyle    stained_glass    vault
-------  ------  ------------  --------  -------------  -------------  -----------------  ----------  ---------------  -------
  0.942  0.6602        0.8629    0.8427         0.9078         0.8762              0.766      0.9536           0.9622   0.9298

Training complete in 16m 56s
Best val Acc: 0.893162

```

## resnet152

BatchSize=32
Workers=4
Epochs=25

```bash
Epoch 11/25
----------
train Loss: 0.1198 Acc: 0.9607
test Loss: 0.1652 Acc: 0.9444

F1 Score:
  altar    apse    bell_tower    column    dome(inner)    dome(outer)    flying_buttress    gargoyle    stained_glass    vault
-------  ------  ------------  --------  -------------  -------------  -----------------  ----------  ---------------  -------
   0.96  0.8738        0.9193    0.9448         0.9014         0.9356             0.9262      0.9853           0.9547   0.9446

Training complete in 79m 41s
Best val Acc: 0.944444
```

## hrnet v2 largest

BatchSize=20
Workers=4
Epochs=25

```bash
Epoch 10/25
----------
train Loss: 0.1009 Acc: 0.9681
test Loss: 0.1793 Acc: 0.9487

F1 Score:
  altar    apse    bell_tower    column    dome(inner)    dome(outer)    flying_buttress    gargoyle    stained_glass    vault
-------  ------  ------------  --------  -------------  -------------  -----------------  ----------  ---------------  -------
 0.9373  0.9091         0.944    0.9458         0.9489          0.936             0.9396      0.9787            0.966   0.9364

Training complete in 221m 59s
Best val Acc: 0.948718
```


## Result Table

**model**|**altar**|**apse**|**bell tower**|**column**|**dome(inner)**|**dome(outer)**|**flying buttress**|**gargoyle**|**stained glass**|**vault**|**performance**
:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:
**baseline**      |0.906     |0.874|0.903 |0.953|0.967|0.937 |0.805 |0.923 |0.990 |0.925|+/-0
**hrnet v1 small**|**0.9353**|0.713|0.8521|0.866|0.922|0.9043|**0.8442**|**0.9437**|0.9728|**0.9254**|-2
**resnet 18**     |**0.942**|0.6602|0.8629|0.8427|0.9078|0.8762|0.766|**0.9536**|**0.9622**|**0.9298**|-2
**resnet 152**    |**0.96**|0.8738|**0.9193**|0.9448|0.9014|0.9356|**0.9262**|**0.9853**|0.9547|**0.9446**|+/-0
**hrnet v2 largest**|**0.9373**|**0.9091**|**0.944**|0.9458|0.9489|0.936|**0.9396**|**0.9787**|0.966|**0.9364**|**+2**

*hrnet v2* beats the baseline model in six of ten classes and the f1 score is remarkable well balanced over all classes. The baseline f1 scores `max - min` is 0.185, whereas *hrnet v2* max distance of the classes is 0.0696.
