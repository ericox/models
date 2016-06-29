# Communication Benchmark Models

This module contains simple tensorflow models that can be used for investigating communication costs
during a distrbuted tensorflow computation.

# Running The Distributed Model

The distributed communication model is implemented in `parbuffer.py`. To run the model to send a variable between
geeker-3 and geeker-4 create run the following:

```bash
# To time the commnication of a variable of size 4096 between geeker-3 and geeker-4

# On geeker-4 (parameter server)
$ python parbuffer.py --variable_size=4096 --batch_size=100 --node_name=ps

# On geeker-3
$ python parbuffer.py --variable_size=4096 --batch_size=100 >> results
```

# Running The Single-Machine Model

To test the baseline for the cheap operation on the variable `tf.round(y)` you can time the code on a single
machine using the `buffer.py` script

```bash
$ python buffer.py --variable_size=4096 --batch_size=100
```

# TODO

 - combine scripts into one module
 - add a parsing script
