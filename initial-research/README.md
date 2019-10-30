i have taken this from https://github.com/fyu/lsun  
image lmdb can be viewed with 
```
conda create -n cv2 python=3.7
conda activate cv2
conda install opencv
pip install lmdb
python data.py view ./
```

[LSUN: Construction of a Large-scale Image Dataset 
using Deep Learning with Humans in the Loop](https://www.yf.io/p/lsun)  

categories:
- bedroom
- bridge
- church_outdoor
- classroom
- conference_room
- dining_room
- kitchen
- living_room
- restaurant
- test
- tower

direct download datasets via http://dl.yf.io/lsun/scenes/
```html
<pre><a href="../">../</a>
<a href="bedroom_train_lmdb.zip">bedroom_train_lmdb.zip</a>                             12-Mar-2017 22:53     43G
<a href="bedroom_val_lmdb.zip">bedroom_val_lmdb.zip</a>                               12-Mar-2017 22:54      4M
<a href="bridge_train_lmdb.zip">bridge_train_lmdb.zip</a>                              12-Mar-2017 23:01     15G
<a href="bridge_val_lmdb.zip">bridge_val_lmdb.zip</a>                                12-Mar-2017 23:02      6M
<a href="categories.txt">categories.txt</a>                                     06-Mar-2019 00:21     110
<a href="church_outdoor_train_lmdb.zip">church_outdoor_train_lmdb.zip</a>                      12-Mar-2017 23:03      2G
<a href="church_outdoor_val_lmdb.zip">church_outdoor_val_lmdb.zip</a>                        12-Mar-2017 23:03      6M
<a href="classroom_train_lmdb.zip">classroom_train_lmdb.zip</a>                           12-Mar-2017 23:04      3G
<a href="classroom_val_lmdb.zip">classroom_val_lmdb.zip</a>                             12-Mar-2017 23:04      6M
<a href="conference_room_train_lmdb.zip">conference_room_train_lmdb.zip</a>                     12-Mar-2017 23:06      4G
<a href="conference_room_val_lmdb.zip">conference_room_val_lmdb.zip</a>                       12-Mar-2017 23:06      5M
<a href="dining_room_train_lmdb.zip">dining_room_train_lmdb.zip</a>                         12-Mar-2017 23:12     11G
<a href="dining_room_val_lmdb.zip">dining_room_val_lmdb.zip</a>                           12-Mar-2017 23:12      5M
<a href="kitchen_train_lmdb.zip">kitchen_train_lmdb.zip</a>                             12-Mar-2017 23:29     33G
<a href="kitchen_val_lmdb.zip">kitchen_val_lmdb.zip</a>                               12-Mar-2017 23:29      5M
<a href="living_room_train_lmdb.zip">living_room_train_lmdb.zip</a>                         12-Mar-2017 23:40     21G
<a href="living_room_val_lmdb.zip">living_room_val_lmdb.zip</a>                           12-Mar-2017 23:40      5M
<a href="restaurant_train_lmdb.zip">restaurant_train_lmdb.zip</a>                          12-Mar-2017 23:46     13G
<a href="restaurant_val_lmdb.zip">restaurant_val_lmdb.zip</a>                            12-Mar-2017 23:47      6M
<a href="test_lmdb.zip">test_lmdb.zip</a>                                      12-Mar-2017 23:52    172M
<a href="tower_train_lmdb.zip">tower_train_lmdb.zip</a>                               12-Mar-2017 23:52     11G
<a href="tower_val_lmdb.zip">tower_val_lmdb.zip</a>                                 12-Mar-2017 23:52      5M
</pre>
```