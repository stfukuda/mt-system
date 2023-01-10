Getting Started
===============

Dependencies
------------

* Python (>=3.8)
* scikit-learn (>=1.1.0)

Installation
------------

It can be installed as follows using pip:

.. code-block:: shell

   pip install -U mt-system

Usage
-----

It learns and predicts like scikit-learn models.

.. code-block:: python

   from mts import MT

   mt = MT(method="mt")

   mt.fit(train_X)

   label = mt.predict(test_X)

   md = mt.mahalanobis(test_X)

or

.. code-block:: python

   from mts import MT

   mt = MT(method="mt")

   label = mt.fit(train_X).predict(test_X)

MT, MTA and SVP methods use the MT model.

.. code-block:: python

   from mts import MT

   mt = MT(method="mt")
   mta = MT(method="mta")
   svp = MT(method="svp")

T(1), T(2), Ta and Tb methods use the T model.

.. code-block:: python

   from mts import T

   t = T(tb=False)  # T(1), T(2), and Ta methods are specified when fitting the model.
   tb = T(tb=True)

   t.fit(train_X, us_idx=us_idx)  # T(1) and T(2) methods.
   t.fit(train_X, us_idx=None)  # Ta method.

RT method use the RT model.

.. code-block:: python

   from mts import RT

   rt = RT()

License
-------

BSD-3-Clause License
