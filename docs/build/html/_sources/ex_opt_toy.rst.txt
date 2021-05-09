Example: Optimization and expand
======================================

In this example, we optimize a tiny octree's output to the
RGB color vector :code:`[0, 1, 0.5]`, for a fixed ray.

We start with data format SH1 and  in the middle switch to SH4 using the :code:`expand(format)` function,
which automatically inserts extra channels as appropriate.
Then we continue to optimize using a manual gradient descent with MSE.
Slowly, the results get closer to the target vector.


.. literalinclude:: ex_opt_toy.py
     :language: python

The output:

.. code::

       GRADIENT DESC
       [0.88920575 0.88920575 0.88920575]
       [0.67369866 0.6859846  0.67984015]
       [0.6194525  0.65586865 0.63762873]
       [0.58019906 0.64437073 0.61214054]
       [0.5475207  0.6409838  0.59386927]
       [0.5188446 0.6420485 0.579674 ]
       [0.49309036 0.64582    0.5681365 ]
       [0.46970066 0.6513118  0.55849427]
       [0.4483344 0.657904  0.5502867]
       [0.42875046 0.66518104 0.54321104]
       Expanding..
       SH4
       [0.4107593 0.6728529 0.5370555]
       [0.3631369  0.71049845 0.5277597 ]
       [0.32639033 0.7405325  0.52003586]
       [0.29751268 0.7646569  0.51378375]
       [0.27432522 0.7842779  0.5088086 ]
       [0.25531954 0.80046684 0.50490075]
       [0.23945224 0.8140159  0.50186735]
       [0.22599061 0.8255081  0.4995423 ]
       [0.21440998 0.83537465 0.4977861 ]
       [0.2043267  0.84393847 0.49648416]
       TARGET
       [0.  1.  0.5]

